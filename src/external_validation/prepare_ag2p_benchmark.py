"""
Generic adapter for AG2P-DISC Kaggle `curated_geno_pheno.rdata` benchmarks.
Replaces the rice210/commonbean-specific scripts: works for any dataset in
the collection where a 'geno' object (accessions x markers, IDs as row
names) and a 'pheno' object (an 'id'-like column + trait columns) share a
common accession-ID scheme.

Usage:
    python src/external_validation/prepare_ag2p_benchmark.py <path-to-curated_geno_pheno.rdata> <dataset_name>

<dataset_name> is a short slug (e.g. "rice210", "commonbean", "soybean")
used to name the output files:
    data/benchmarks/learnable/processed/genotype_external_<dataset_name>.csv
    data/benchmarks/learnable/processed/phenotype_external_<dataset_name>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyreadr

BASE_DIR = Path(__file__).resolve().parent.parent.parent # Root directory of the project.
OUT_DIR = BASE_DIR / "data" / "benchmarks" / "learnable" / "processed" # Directory to save processed benchmark files.

ID_COLUMN_CANDIDATES = ("id", "clone_id", "accession", "genotype", "line")

# Helper function to normalize clone IDs.
def _normalize_clone_id(raw_id: object) -> str: # Normalize the clone ID by trimming/collapsing whitespace and converting to uppercase.
    """Trim/collapse whitespace and uppercase, matching src/etl_pipeline.py's convention."""
    return " ".join(str(raw_id).split()).upper()

# Helper function to find an object in the RData result dict by keyword.
def _find_object(result: dict[str, pd.DataFrame], keyword: str) -> tuple[str, pd.DataFrame]: 
    """Find the object whose name contains `keyword`, case-insensitively."""
    matches = [(name, df) for name, df in result.items() if keyword in name.lower()]
    if not matches:
        raise ValueError(
            f"No object with {keyword!r} in its name found; objects present: {list(result.keys())}"
        )
    # Prefer an exact match (e.g. 'geno' over 'genotype_info'), else the largest object.
    exact = [m for m in matches if m[0].lower() == keyword] 
    return exact[0] if exact else max(matches, key=lambda m: m[1].size) 

# Helper function to find the ID columns in a DataFrame.
def _find_id_column(df: pd.DataFrame) -> str | None: 
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None

# Main function to prepare genotype and phenotype DataFrames from an RData file.
def prepare(rdata_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:  
    result = pyreadr.read_r(str(rdata_path))
    geno_name, geno_df = _find_object(result, "geno")
    pheno_name, pheno_df = _find_object(result, "pheno")
    geno_df, pheno_df = geno_df.copy(), pheno_df.copy()
    print(f"Using {geno_name!r} as genotype ({geno_df.shape}) and {pheno_name!r} as phenotype ({pheno_df.shape})")
    # Normalize genotype clone IDs and handle missing ID column.
    geno_id_column = _find_id_column(geno_df)
    if geno_id_column:
        geno_df["clone_id"] = geno_df[geno_id_column].map(_normalize_clone_id)
        geno_df = geno_df.drop(columns=geno_id_column)
    else:
        # No explicit id column: the accession IDs are the R row names, which
        # pyreadr surfaces as the pandas index.
        geno_df.insert(0, "clone_id", [_normalize_clone_id(i) for i in geno_df.index])
        geno_df = geno_df.reset_index(drop=True)
    marker_columns = [c for c in geno_df.columns if c != "clone_id"] # All columns except the clone ID are considered marker columns.
    for column in marker_columns:
        geno_df[column] = pd.to_numeric(geno_df[column], errors="coerce") # Convert marker columns to numeric, coercing errors to NaN.
    geno_df = geno_df.groupby("clone_id", as_index=False)[marker_columns].mean() # Aggregate marker columns by clone ID, taking the mean for duplicates.

    pheno_id_column = _find_id_column(pheno_df)
    if pheno_id_column:
        pheno_df["clone_id"] = pheno_df[pheno_id_column].map(_normalize_clone_id)
        pheno_df = pheno_df.drop(columns=pheno_id_column)
    else:
        pheno_df.insert(0, "clone_id", [_normalize_clone_id(i) for i in pheno_df.index])
        pheno_df = pheno_df.reset_index(drop=True)
    trait_columns = [c for c in pheno_df.columns if c != "clone_id"]
    for column in trait_columns:
        pheno_df[column] = pd.to_numeric(pheno_df[column], errors="coerce") # Convert trait columns to numeric, coercing errors to NaN.
    pheno_df = pheno_df.drop_duplicates(subset="clone_id", keep="first")
    pheno_df = pheno_df[["clone_id", *trait_columns]]
    # Identify overlapping clone IDs between genotype and phenotype tables.
    overlap = set(geno_df["clone_id"]) & set(pheno_df["clone_id"]) # Compute the intersection of clone IDs between genotype and phenotype tables.
    print(f"Genotype clones: {len(geno_df)}, Phenotype clones: {len(pheno_df)}, overlap: {len(overlap)}")
    print(f"Marker columns: {len(marker_columns)}; trait columns: {trait_columns}")
    if not overlap:
        raise ValueError(
            "No overlapping clone IDs between genotype and phenotype tables; "
            "inspect the raw .rdata object names/columns manually."
        )

    return geno_df, pheno_df

# Main entry point for the script. Parses command-line arguments and prepares the genotype and phenotype CSVs.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rdata_path", help="Path to curated_geno_pheno.rdata")
    parser.add_argument("dataset_name", help="Short slug used to name the output CSVs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    genotype, phenotype = prepare(args.rdata_path)
    geno_out = OUT_DIR / f"genotype_external_{args.dataset_name}.csv"
    pheno_out = OUT_DIR / f"phenotype_external_{args.dataset_name}.csv"
    genotype.to_csv(geno_out, index=False)
    phenotype.to_csv(pheno_out, index=False)
    print(f"Wrote {geno_out} and {pheno_out}")


if __name__ == "__main__":
    main()
