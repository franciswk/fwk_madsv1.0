"""Calculate the five phenotype indices used by the notebook pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Columns averaged into the Disease Resistance Index (DRI). Each is expected to
# already be on a 0-1 scale; non-numeric entries (e.g. text ratings) are skipped.
RESISTANCE_COLUMNS = (
    "fusarium_wilt_resistance",
    "soil_rot_resistance",
    "southern_rkn_resistance",
    "fwilt_measured",
    "ssr_measured",
    "srkn_measured",
    "grkn_measured",
)

# Helper function to bound numeric values within a specified range and scale to 0-1.
def _bounded(value: object, lower: float, upper: float) -> float | None:
    # Missing or non-numeric values return None ("unavailable") rather than 0,
    # so callers can fall back to another source or exclude them from averages.
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    # Min-max scale [lower, upper] to [0, 1], clipping out-of-range values.
    return float(np.clip((float(numeric) - lower) / (upper - lower), 0, 1))

# Helper function to score free-text fields based on keyword matches.
def _text_score(value: object, scores: dict[str, float], default: float = 0.5) -> float: # 0.5 is giving a neutral midpoint when no keywords match
    # Case-insensitive substring match against a free-text field.
    text = str(value).strip().lower()
    # The first matching phrase wins, so list more specific phrases first
    # (e.g. "deep orange" before "orange").
    for phrase, score in scores.items():
        if phrase in text:
            return score
    # Neutral midpoint when nothing matches (including missing values).
    return default

# Calculate the Eating Quality Index (EQI) based on sweetness, color, and moistness.
def calculate_eqi(row: pd.Series) -> float:
    """Equal-weight sweetness, deep flesh color, and moistness proxies."""
    # Sweetness: prefer measured starch (3-18% scaled to 0-1); otherwise use
    # keywords in the market category.
    starch = _bounded(row.get("pct_starch_measured"), 3, 18) # 3, 18 reflect the typical biological and phenotypic range of starch content (expressed on a fresh weight basis) found across diverse sweetpotato breeding populations and germplasm collections.
    sweetness = starch if starch is not None else _text_score( # should be replaced with sensory panel taste results if available
        row.get("market_category"), {"sweet": 0.75, "sucrose": 0.75}
    )

    # Color: prefer measured beta-carotene (0-0.6 scaled to 0-1); otherwise
    # rank the described flesh color, with deeper orange scoring highest.
    beta_carotene = _bounded(row.get("beta_carotene_measured"), 0, 0.6) # 0.6 Represents the high-end ceiling for beta-carotene concentration in elite orange-fleshed sweetpotato (OFSP) breeding lines and cultivars
    color = beta_carotene if beta_carotene is not None else _text_score(
        row.get("flesh_color"),
        {"deep orange": 1.0, "orange": 0.75, "purple": 0.65, "yellow": 0.35, "white": 0.1},
    )

    # Moistness: a triangular score that peaks at 22% dry matter and drops
    # linearly to 0 at 10% or 34%. Without dry-matter data, use market-category
    # keywords instead.
    dry_matter = pd.to_numeric(row.get("pct_dm_measured"), errors="coerce")
    moistness = (
        float(np.clip(1 - abs(float(dry_matter) - 22) / 12, 0, 1))
        if pd.notna(dry_matter)
        else _text_score(row.get("market_category"), {"moist": 0.85, "dry": 0.2})
    )
    # The three components are always present (via fallbacks), so EQI is
    # their equal-weight mean.
    return float(np.mean([sweetness, color, moistness]))

# Calculate the Disease Resistance Index (DRI) based on available resistance scores.
def calculate_dri(row: pd.Series) -> float:
    """Average available disease-resistance scores, all normalized to 0-1."""
    # Collect only the resistance values that are present and numeric.
    scores = []
    for column in RESISTANCE_COLUMNS:
        score = _bounded(row.get(column), 0, 1)
        if score is not None:
            scores.append(score)
    # Missing values are excluded rather than counted as 0; a row with no
    # resistance data at all scores 0.
    return float(np.mean(scores)) if scores else 0.0

# Calculate the Agronomic Yield Index (AYI) based on yield and maturity window.
def calculate_ayi(row: pd.Series) -> float:
    """Combine yield and an agronomic maturity-window preference."""
    # Yield of 0.3-1.8 kg is scaled to 0-1 (higher is better).
    yield_score = _bounded(row.get("yield_kg_measured"), 0.3, 1.8)
    # Days after planting (DAP) to maturity, scaled within typical ranges.
    maturity_min = _bounded(row.get("maturity_dap_min"), 90, 120)
    maturity_max = _bounded(row.get("maturity_dap_max"), 120, 140)
    # Invert both endpoints so earlier maturity scores higher; both are
    # required to form a maturity score.
    maturity_score = None
    if maturity_min is not None and maturity_max is not None:
        maturity_score = float(np.mean([1 - maturity_min, 1 - maturity_max]))
    # Average whichever components are available; 0 if neither is.
    scores = [score for score in (yield_score, maturity_score) if score is not None]
    return float(np.mean(scores)) if scores else 0.0

# Calculate the Parental Value Score (PVS) as a weighted combination of DRI, AYI, and EQI.
def calculate_parental_value(row: pd.Series, dri: float, eqi: float, ayi: float) -> float:
    """Calculate parental value as a weighted proxy from computed indices."""
    # Breeding-parent weighting: disease resistance 40%, agronomics 40%, eating quality 20% - reflects the relative importance assigned to each component in the overall parental value score.
    return float(dri * 0.4 + ayi * 0.4 + eqi * 0.2)

# Calculate all indices for a given DataFrame, including base and composite indices.
def calculate_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the five project indices recalculated."""
    # Work on a copy so the caller's DataFrame is not modified.
    scored = df.copy()
    # Base indices are computed row by row and rounded to 6 decimals.
    scored["dri"] = scored.apply(calculate_dri, axis=1).round(6)
    scored["eqi"] = scored.apply(calculate_eqi, axis=1).round(6)
    scored["ayi"] = scored.apply(calculate_ayi, axis=1).round(6)
    # Composite indices are built from the already-rounded base indices:
    # market_index is an unweighted mean; parental_value is weighted.
    scored["market_index"] = scored[["dri", "eqi", "ayi"]].mean(axis=1).round(6)
    scored["parental_value"] = scored.apply(
        lambda row: round(calculate_parental_value(row, row["dri"], row["eqi"], row["ayi"]), 6),
        axis=1,
    )
    return scored

# Score a single CSV file by recalculating all indices and writing the output.
def score_file(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Read a phenotype CSV, recalculate indices, and write the scored CSV."""
    scored = calculate_indices(pd.read_csv(input_path))
    # Create the output folder if needed, and write without the pandas index column.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored