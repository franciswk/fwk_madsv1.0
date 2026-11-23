"""
Simulate data/raw/raw_genotypes.csv from data/raw/raw_phenotypes.csv alone.

The phenotype file is cleaned and scored in memory with the project ETL, so the
embedded signal always matches the current index formulas. The 10,000-clone
roster is the labeled clones plus seeded, generated breeding-line candidates.

Simulation:
  1. Realistic marker properties: each SNP gets a minor allele frequency (MAF)
     drawn from a U-shaped Beta distribution (like real SNP spectra), and clones
     are drawn from population-structured subgroups ("breeding programs") with
     their own small per-marker allele-frequency drift (Fst-like structure).
  2. Linkage disequilibrium: markers are grouped into blocks; within a block,
     a clone's genotype at later markers is frequently copied from the block's
     anchor marker, mimicking haplotype/LD structure instead of independent SNPs.
  3. Causal markers with real signal: for each modeled phenotype target, a
     random subset of markers is chosen as "causal". For labeled clones with a
     known phenotype value, that marker's allele frequency is nudged up/down
     based on the clone's standardized phenotype before drawing the (still
     stochastic, Binomial/HWE) genotype call. This embeds a moderate, noisy,
     discretized dose-response relationship - the same kind of signal real
     genomic prediction relies on - without deterministically leaking the
     target. Unlabeled breeding candidates get no such nudge (their phenotype
     is exactly what the model should predict).

Run from the project root (backs up any existing genotype file first), then
re-run the ETL and analysis:
    python -m src.data_generation.simulate_realistic_genotypes
    python src/etl_pipeline.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.etl_pipeline import extract_phenotype, transform_phenotype

BASE_DIR = Path(__file__).resolve().parents[2]
PHENOTYPE_PATH = BASE_DIR / "data" / "raw" / "raw_phenotypes.csv"
GENOTYPE_PATH = BASE_DIR / "data" / "raw" / "raw_genotypes.csv"

SEED = 7
N_CLONES = 10_000
# Candidate breeding programs and their share of the unlabeled roster.
CANDIDATE_PROGRAMS = {"LSU": 0.2214, "NCSU": 0.1807, "CIP": 0.1555, "UGA": 0.1531, "UF": 0.1463, "USDA": 0.1430}
N_MARKERS = 500
LD_BLOCK_SIZE = 10
LD_COPY_PROBABILITY = 0.65
CAUSAL_MARKERS_PER_TARGET = 25
EFFECT_STRENGTH_RANGE = (0.25, 0.45)
N_PROGRAMS = 6
PROGRAM_DRIFT_STD = 0.05

# Computed indices with embedded genetic signal.
CONDITIONED_TARGETS = ["market_index", "dri", "eqi", "parental_value", "ayi"]
TARGET_SOURCE_COLUMNS: dict[str, str] = {"parental_value": "parental_value"}

# Messiness applied after simulation, for the ETL step to clean.
MISSING_CALL_RATE = 0.003
N_MALFORMED_CELLS = 60
N_DUPLICATE_ROWS = 5


def _messify_clone_id(clone_id: str, rng: np.random.Generator) -> str:
    roll = rng.random()
    if roll < 0.15:
        return clone_id.lower()
    if roll < 0.25:
        return clone_id.title()
    if roll < 0.35:
        return f"  {clone_id}  "
    if roll < 0.40:
        return clone_id.replace(" ", "  ")
    return clone_id


def _zscore(series: pd.Series) -> pd.Series:
    standard_deviation = series.std(ddof=0)
    if standard_deviation == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / standard_deviation


def build_clone_roster(labeled_ids: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Combine labeled clone_ids with generated breeding-line candidates (e.g. UF_00001)."""
    n_unlabeled = N_CLONES - len(labeled_ids)
    if n_unlabeled < 1:
        raise ValueError(f"More labeled clones ({len(labeled_ids)}) than roster slots ({N_CLONES}).")
    names = list(CANDIDATE_PROGRAMS)
    shares = np.array(list(CANDIDATE_PROGRAMS.values()))
    programs = rng.choice(names, size=n_unlabeled, p=shares / shares.sum()).tolist()
    unlabeled_ids = [f"{program}_{number:05d}" for number, program in enumerate(programs, start=1)]

    all_ids = labeled_ids + unlabeled_ids
    unique_unlabeled_programs = sorted(set(programs))[:N_PROGRAMS]
    labeled_programs = [rng.choice(unique_unlabeled_programs) for _ in labeled_ids]

    return pd.DataFrame(
        {
            "clone_id": all_ids,
            "program": labeled_programs + programs,
            "is_labeled": [True] * len(labeled_ids) + [False] * len(unlabeled_ids),
        }
    )


def simulate_marker_matrix(
    roster: pd.DataFrame,
    labeled_targets: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate MAF/population-structure/LD genotypes with embedded causal signal."""
    n_clones = len(roster)
    programs = roster["program"].to_numpy()
    unique_programs = sorted(roster["program"].unique())
    program_index = {p: i for i, p in enumerate(unique_programs)}
    program_idx_per_clone = np.array([program_index[p] for p in programs])

    base_maf = np.clip(rng.beta(0.4, 0.4, size=N_MARKERS), 0.05, 0.95)
    program_maf = np.clip(
        base_maf[None, :] + rng.normal(0, PROGRAM_DRIFT_STD, size=(len(unique_programs), N_MARKERS)),
        0.02,
        0.98,
    )

    # Allele-frequency matrix per clone x marker, before causal-signal nudging.
    allele_freq = program_maf[program_idx_per_clone, :]

    # Embed causal, noisy signal for labeled clones per target.
    label_position = roster.index[roster["is_labeled"]].to_numpy()
    all_causal_markers: set[int] = set()
    for target in CONDITIONED_TARGETS:
        source_column = TARGET_SOURCE_COLUMNS.get(target, target)
        if source_column not in labeled_targets.columns:
            continue
        y = labeled_targets[source_column]
        y_std_full = np.zeros(n_clones)
        non_null_mask = y.notna()
        if non_null_mask.sum() >= 2:
            z = _zscore(y.loc[non_null_mask]).to_numpy()
            positions = label_position[np.where(non_null_mask.to_numpy())[0]]
            y_std_full[positions] = z

        causal_markers = rng.choice(N_MARKERS, size=CAUSAL_MARKERS_PER_TARGET, replace=False)
        slopes = rng.uniform(*EFFECT_STRENGTH_RANGE, size=CAUSAL_MARKERS_PER_TARGET) * rng.choice(
            [-1.0, 1.0], size=CAUSAL_MARKERS_PER_TARGET
        )
        allele_freq[:, causal_markers] += np.outer(y_std_full, slopes)
        all_causal_markers.update(causal_markers.tolist())

    allele_freq = np.clip(allele_freq, 0.02, 0.98)
    dosage = rng.binomial(2, allele_freq).astype(float)

    # Linkage disequilibrium: copy the block anchor marker's call for a random subset of
    # clones, but never overwrite a causal marker's own embedded signal.
    for block_start in range(0, N_MARKERS, LD_BLOCK_SIZE):
        block_cols = range(block_start, min(block_start + LD_BLOCK_SIZE, N_MARKERS))
        anchor_col = block_start
        for col in block_cols:
            if col == anchor_col or col in all_causal_markers:
                continue
            copy_mask = rng.random(n_clones) < LD_COPY_PROBABILITY
            dosage[copy_mask, col] = dosage[copy_mask, anchor_col]

    marker_columns = [f"SNP_{i + 1:04d}" for i in range(N_MARKERS)]
    genotype_df = pd.DataFrame(dosage, columns=marker_columns)
    genotype_df.insert(0, "clone_id", roster["clone_id"].to_numpy())
    return genotype_df


def apply_messiness(genotype_df: pd.DataFrame, rng: np.random.Generator, n_labeled: int) -> pd.DataFrame:
    """Re-apply the same missing-call / malformed-token / duplicate-row noise as before."""
    df = genotype_df.copy()
    marker_columns = [c for c in df.columns if c != "clone_id"]

    # Messy casing/whitespace variants for a subset of the labeled clone_id rows.
    messy_mask = rng.random(n_labeled) < 0.4
    labeled_ids = df.loc[: n_labeled - 1, "clone_id"]
    df.loc[: n_labeled - 1, "clone_id"] = [
        _messify_clone_id(cid, rng) if messy else cid for cid, messy in zip(labeled_ids, messy_mask)
    ]

    marker_values = df[marker_columns].to_numpy(dtype=float, copy=True)
    missing_mask = rng.random(marker_values.shape) < MISSING_CALL_RATE
    marker_values[missing_mask] = np.nan
    df[marker_columns] = marker_values

    malformed_rows = rng.choice(df.index, size=N_MALFORMED_CELLS, replace=False)
    malformed_cols = rng.choice(marker_columns, size=N_MALFORMED_CELLS, replace=True)
    malformed_tokens = rng.choice(["NA", "./.", "no_call"], size=N_MALFORMED_CELLS)
    df = df.astype(object)
    for row_idx, col_name, token in zip(malformed_rows, malformed_cols, malformed_tokens):
        df.at[row_idx, col_name] = token

    dup_positions = rng.choice(df.index[:n_labeled], size=min(N_DUPLICATE_ROWS, n_labeled), replace=False)
    duplicate_rows = df.loc[dup_positions].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Clean and score the raw phenotypes with the project ETL (current index formulas).
    labeled_targets, _ = transform_phenotype(extract_phenotype(PHENOTYPE_PATH))
    labeled_targets = labeled_targets.set_index("clone_id")
    labeled_ids = labeled_targets.index.tolist()

    roster = build_clone_roster(labeled_ids, rng)
    labeled_targets_aligned = labeled_targets.reindex(roster.loc[roster["is_labeled"], "clone_id"])
    labeled_targets_aligned = labeled_targets_aligned.reset_index(drop=True)

    genotype_df = simulate_marker_matrix(roster, labeled_targets_aligned, rng)
    messy_genotype_df = apply_messiness(genotype_df, rng, n_labeled=len(labeled_ids))

    if GENOTYPE_PATH.exists():
        backup_dir = GENOTYPE_PATH.parent / f"backup_{datetime.now():%Y%m%d_%H%M%S}"
        backup_dir.mkdir(parents=True)
        shutil.copy2(GENOTYPE_PATH, backup_dir / GENOTYPE_PATH.name)
        print(f"Previous genotype file backed up to: {backup_dir}")

    GENOTYPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    messy_genotype_df.to_csv(GENOTYPE_PATH, index=False)

    print(f"Simulated genotype rows: {len(messy_genotype_df)} ({len(labeled_ids)} labeled clones)")
    print(f"Causal markers embedded per target: {CAUSAL_MARKERS_PER_TARGET} for {CONDITIONED_TARGETS}")
    print(f"Wrote {GENOTYPE_PATH}")


if __name__ == "__main__":
    main()
