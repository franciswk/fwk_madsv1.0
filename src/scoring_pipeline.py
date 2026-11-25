"""Calculate the five phenotype indices used by the notebook pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Measured columns for the Disease Resistance Index (DRI), one per disease.
# Values are expected on a 0-1 scale; missing or non-numeric entries are skipped.
DISEASE_COLUMNS = {
    "fusarium_wilt": "fwilt_measured",
    "soil_rot": "ssr_measured",
    "southern_rkn": "srkn_measured",
    "guava_rkn": "grkn_measured",
}

# Helper function to bound numeric values within a specified range and scale to 0-1.
def _bounded(value: object, lower: float, upper: float) -> float | None:
    # Missing or non-numeric values return None ("unavailable") rather than 0,
    # so callers can exclude them from averages.
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    # Min-max scale [lower, upper] to [0, 1], clipping out-of-range values.
    return float(np.clip((float(numeric) - lower) / (upper - lower), 0, 1))

# Calculate the Eating Quality Index (EQI) based on sweetness, color, and moistness.
def calculate_eqi(row: pd.Series) -> float:
    """Equal-weight mean of the measured sweetness, color, and moistness scores."""
    # Sweetness: measured starch, 3-18% scaled to 0-1. 3 and 18 reflect the
    # typical range of starch content (fresh weight basis) across diverse
    # sweetpotato breeding populations and germplasm collections.
    sweetness = _bounded(row.get("pct_starch_measured"), 3, 18)

    # Color: measured beta-carotene, 0-0.6 scaled to 0-1. 0.6 is the high-end
    # ceiling for elite orange-fleshed sweetpotato (OFSP) lines and cultivars.
    color = _bounded(row.get("beta_carotene_measured"), 0, 0.6)

    # Moistness: a triangular score that peaks at 22% dry matter and drops
    # linearly to 0 at 10% or 34%.
    dry_matter = pd.to_numeric(row.get("pct_dm_measured"), errors="coerce")
    moistness = (
        float(np.clip(1 - abs(float(dry_matter) - 22) / 12, 0, 1))
        if pd.notna(dry_matter)
        else None
    )
    # Average whichever components are measured; NaN (unscorable) if none are.
    scores = [score for score in (sweetness, color, moistness) if score is not None]
    return float(np.mean(scores)) if scores else np.nan

# Calculate the Disease Resistance Index (DRI) based on available resistance scores.
def calculate_dri(row: pd.Series) -> float:
    """Mean of the measured 0-1 resistance scores available for this row."""
    scores = []
    for column in DISEASE_COLUMNS.values():
        score = _bounded(row.get(column), 0, 1)
        if score is not None:
            scores.append(score)
    # Diseases with no data are excluded rather than counted as 0; a row with
    # no resistance data at all is NaN (unscorable), not the worst score.
    return float(np.mean(scores)) if scores else np.nan

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
    # Average whichever components are available; NaN (unscorable) if neither is.
    scores = [score for score in (yield_score, maturity_score) if score is not None]
    return float(np.mean(scores)) if scores else np.nan

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
    # market_index is an unweighted mean; parental_value is weighted. Both are
    # NaN when any base index is NaN, rather than averaging around the gap.
    scored["market_index"] = scored[["dri", "eqi", "ayi"]].mean(axis=1, skipna=False).round(6)
    scored["parental_value"] = scored.apply(
        lambda row: round(calculate_parental_value(row, row["dri"], row["eqi"], row["ayi"]), 6),
        axis=1,
    )
    return scored