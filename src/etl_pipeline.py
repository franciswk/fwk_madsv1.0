"""
ETL pipeline for the SNP genomic-prediction inputs.

Extract:   read the raw, messy CSVs (data/raw/raw_phenotypes.csv
           and data/raw/raw_genotypes.csv).
Transform: normalize clone_id keys, coerce mixed-type/percentage/missing-token
           values to proper numeric dtypes, and resolve duplicate rows.
Load:      write clean, model-ready CSVs to data/processed/ that satisfy the
           strict schema `capstone_pipeline.validate_inputs` expects (unique,
           non-null clone_id; fully numeric marker columns; numeric target).

Run as a script to (re)generate the processed files:
    python src/etl_pipeline.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .scoring_pipeline import calculate_indices
except ImportError:
    from scoring_pipeline import calculate_indices

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PHENOTYPE_PATH = BASE_DIR / "data" / "raw" / "raw_phenotypes.csv"
RAW_GENOTYPE_PATH = BASE_DIR / "data" / "raw" / "raw_genotypes.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_PHENOTYPE_PATH = PROCESSED_DIR / "phenotype_clean.csv"
CLEAN_GENOTYPE_PATH = PROCESSED_DIR / "genotype_clean.csv"
ETL_REPORT_PATH = PROCESSED_DIR / "etl_report.json"

TARGET_COLUMN = "market_index"

# Alternate missing-value tokens seen in the raw exports, beyond pandas' defaults.
MISSING_TOKENS = ["NA", "N/A", "unknown", "--", "TBD", "./.", "no_call"]

NUMERIC_PHENOTYPE_COLUMNS = [
    "release_year",
    "maturity_dap_min",
    "maturity_dap_max",
    "dry_matter_pct_literature",
    "fusarium_wilt_resistance",
    "soil_rot_resistance",
    "southern_rkn_resistance",
    "yield_kg_measured",
    "pct_dm_measured",
    "pct_starch_measured",
    "beta_carotene_measured",
    "fwilt_measured",
    "ssr_measured",
    "srkn_measured",
    "grkn_measured",
]

# Variables that count as "measured"; n_measured_sources is derived from these.
MEASURED_TRAIT_COLUMNS = [
    "yield_kg_measured",
    "pct_dm_measured",
    "pct_starch_measured",
    "beta_carotene_measured",
    "fwilt_measured",
    "ssr_measured",
    "srkn_measured",
    "grkn_measured",
]

STRING_PHENOTYPE_COLUMNS = [
    "variety",
    "origin",
    "skin_color",
    "flesh_color",
    "market_category",
]


def normalize_clone_id(series: pd.Series) -> pd.Series:
    """Collapse whitespace/casing variants so join keys line up (e.g. ' bonita ' -> 'BONITA')."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
        .replace("", pd.NA)
    )


def _strip_percent_sign(value: object) -> object:
    if isinstance(value, str) and value.strip().endswith("%"):
        return value.strip().rstrip("%")
    return value


# --- Extract -----------------------------------------------------------------


def extract_phenotype(path: str | Path = RAW_PHENOTYPE_PATH) -> pd.DataFrame:
    """Read the raw commercial phenotype CSV, recognizing alternate missing tokens."""
    return pd.read_csv(path, na_values=MISSING_TOKENS, keep_default_na=True)


def extract_genotype(path: str | Path = RAW_GENOTYPE_PATH) -> pd.DataFrame:
    """Read the raw SNP genotype CSV, recognizing alternate missing/no-call tokens."""
    return pd.read_csv(path, na_values=MISSING_TOKENS, keep_default_na=True, low_memory=False)


# --- Transform -----------------------------------------------------------------


def transform_phenotype(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the phenotype table into a strict clone_id/market_index schema."""
    report: dict = {}
    df = df.copy()

    df["clone_id"] = normalize_clone_id(df["clone_id"])
    if df["clone_id"].isna().any():
        raise ValueError("Phenotype input contains missing or blank clone_id values.")

    for column in STRING_PHENOTYPE_COLUMNS:
        df[column] = df[column].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        df.loc[df[column].isin(["nan", ""]), column] = np.nan

    for column in ["pct_dm_measured", "pct_starch_measured"]:
        df[column] = df[column].apply(_strip_percent_sign)

    for column in NUMERIC_PHENOTYPE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Documentation level of each record: how many measured variables it actually has.
    # Computed here, before any dedupe, so the best-documented duplicate wins below.
    df["n_measured_sources"] = df[MEASURED_TRAIT_COLUMNS].notna().sum(axis=1).astype(int)
    n_before_exact_dedupe = len(df)
    df = df.drop_duplicates()
    report["exact_duplicate_rows_dropped"] = n_before_exact_dedupe - len(df)

    n_before_key_dedupe = len(df)
    df = df.sort_values("n_measured_sources", ascending=False, kind="stable")
    df = df.drop_duplicates(subset="clone_id", keep="first")
    df = df.sort_index()

    df = calculate_indices(df)
    report["duplicate_clone_id_rows_resolved"] = n_before_key_dedupe - len(df)

    missing_target = df[TARGET_COLUMN].isna().sum()
    if missing_target:
        df = df.dropna(subset=[TARGET_COLUMN])
    report["rows_dropped_missing_target"] = int(missing_target)

    report["final_row_count"] = len(df)
    assert df["clone_id"].is_unique, "clone_id must be unique after transform"
    assert not df["clone_id"].isna().any(), "clone_id must not contain nulls after transform"

    return df.reset_index(drop=True), report


def transform_genotype(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the SNP genotype matrix into a strict clone_id + numeric-marker schema."""
    report: dict = {}
    df = df.copy()

    df["clone_id"] = normalize_clone_id(df["clone_id"])
    if df["clone_id"].isna().any():
        raise ValueError("Genotype input contains missing or blank clone_id values.")
    marker_columns = [c for c in df.columns if c != "clone_id"]

    non_numeric_cells_before = 0
    for column in marker_columns:
        coerced = pd.to_numeric(df[column], errors="coerce")
        non_numeric_cells_before += int(coerced.isna().sum() - df[column].isna().sum())
        df[column] = coerced
    report["malformed_marker_cells_coerced_to_nan"] = max(non_numeric_cells_before, 0)
    df = df.copy()  # de-fragment after per-column coercion, before drop_duplicates/groupby

    n_before_exact_dedupe = len(df)
    df = df.drop_duplicates()
    report["exact_duplicate_rows_dropped"] = n_before_exact_dedupe - len(df)

    n_before_key_dedupe = len(df)
    df = df.groupby("clone_id", as_index=False)[marker_columns].mean()
    report["duplicate_clone_id_rows_merged"] = n_before_key_dedupe - len(df)

    report["missing_marker_calls"] = int(df[marker_columns].isna().sum().sum())
    report["final_row_count"] = len(df)
    report["marker_count"] = len(marker_columns)

    assert df["clone_id"].is_unique, "clone_id must be unique after transform"
    assert df[marker_columns].apply(pd.api.types.is_numeric_dtype).all(), "all markers must be numeric"

    return df, report


# --- Load -----------------------------------------------------------------


def load_clean(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# --- Orchestration -----------------------------------------------------------------


def run_etl(
    raw_phenotype_path: str | Path = RAW_PHENOTYPE_PATH,
    raw_genotype_path: str | Path = RAW_GENOTYPE_PATH,
    clean_phenotype_path: str | Path = CLEAN_PHENOTYPE_PATH,
    clean_genotype_path: str | Path = CLEAN_GENOTYPE_PATH,
    etl_report_path: str | Path | None = None,
) -> dict:
    """Run ETL; default the report location to the clean phenotype directory."""
    raw_phenotype = extract_phenotype(raw_phenotype_path)
    raw_genotype = extract_genotype(raw_genotype_path)

    clean_phenotype, phenotype_report = transform_phenotype(raw_phenotype)
    clean_genotype, genotype_report = transform_genotype(raw_genotype)

    load_clean(clean_phenotype, clean_phenotype_path)
    load_clean(clean_genotype, clean_genotype_path)

    common_ids = set(clean_phenotype["clone_id"]) & set(clean_genotype["clone_id"])
    report = {
        "phenotype": {"rows_in": len(raw_phenotype), **phenotype_report},
        "genotype": {"rows_in": len(raw_genotype), **genotype_report},
        "overlapping_clone_ids_after_join": len(common_ids),
    }
    report_path = (
        Path(etl_report_path) if etl_report_path is not None
        else Path(clean_phenotype_path).parent / "etl_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "phenotype": clean_phenotype,
        "genotype": clean_genotype,
        "report": report,
    }


def main() -> None:
    result = run_etl()
    print(json.dumps(result["report"], indent=2))
    print(f"Clean phenotype written to: {CLEAN_PHENOTYPE_PATH}")
    print(f"Clean genotype written to:  {CLEAN_GENOTYPE_PATH}")


if __name__ == "__main__":
    main()
