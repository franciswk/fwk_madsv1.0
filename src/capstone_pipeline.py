"""Standalone helpers for the notebook-first sweetpotato capstone.

Multi-target genomic prediction: trains on the labeled (commercial +
synthetic) clones with known phenotypes and predicts each target trait for
every genotyped clone, including the unlabeled breeding candidates.

Run directly to (re)generate outputs/ from data/processed/:
    python src/capstone_pipeline.py                              # full ranking pipeline (all targets)
    python src/capstone_pipeline.py --compare --target market_index  # Elastic Net vs Ridge/MLP comparison
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Primary target used when no explicit validation target is supplied.
TARGET_COLUMN = "market_index"

# Computed indices to predict independently. Measured traits remain inputs to
# scoring and are not genomic prediction targets.
TARGET_COLUMNS = [
    "market_index",
    "dri",
    "eqi",
    "parental_value",
    "ayi",
]

# Computed index columns are the model targets after ETL scoring.
TARGET_SOURCE_COLUMNS = {
    "market_index": "market_index",
    "dri": "dri",
    "eqi": "eqi",
    "parental_value": "parental_value",
    "ayi": "ayi",
}

# A target is only modeled if at least this many labeled clones have a
# non-null value: with ~500 SNP markers, fitting on fewer risks severe
# overfitting even with regularization.
MIN_SAMPLES_PER_TARGET = 15

# Cap the number of SNP markers fed to the model so the sample:feature ratio
# stays as favorable as possible for a small labeled set.
MAX_SELECTED_MARKERS = 50

# Elastic Net is the primary genomic prediction model: marker selection and
# combined L1/L2 regularization control overfitting in the small labeled set.
MODEL_FAMILY = "elastic_net"
MODEL_FAMILIES = ("elastic_net", "ridge", "mlp_baseline")

# Never select more markers than this many samples-per-marker would allow;
# a lower ratio than this is a strong overfitting signal for a linear model.
MIN_SAMPLES_PER_MARKER = 3

MAX_CV_FOLDS = 5
MIN_CV_FOLDS = 3
DEFAULT_SEED = 42

# Default locations when run as a script (see module docstring); importers
# should keep passing explicit paths.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PHENOTYPE_PATH = BASE_DIR / "data" / "processed" / "phenotype_clean.csv"
DEFAULT_GENOTYPE_PATH = BASE_DIR / "data" / "processed" / "genotype_clean.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

# Functions for loading and validating input data.
def load_inputs(
    phenotype_path: str | Path,
    genotype_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load phenotype and genotype CSV inputs."""
    return pd.read_csv(phenotype_path), pd.read_csv(genotype_path)

# Validate the inputs and return an alignment summary.
def validate_inputs(
    phenotype: pd.DataFrame,
    genotype: pd.DataFrame,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Validate the requested targets and return their alignment summary."""
    target_columns = targets if targets is not None else [TARGET_COLUMN]
    if not target_columns:
        raise ValueError("Input validation failed: at least one target is required.")
    phenotype_required = {"clone_id", *target_columns}
    missing_phenotype_columns = sorted(phenotype_required - set(phenotype.columns))
    missing_genotype_columns = sorted({"clone_id"} - set(genotype.columns))
    schema_errors = [
        *(
            ["Phenotype input is missing: " + ", ".join(missing_phenotype_columns)]
            if missing_phenotype_columns
            else []
        ),
        *(
            ["Genotype input is missing: " + ", ".join(missing_genotype_columns)]
            if missing_genotype_columns
            else []
        ),
    ]
    if schema_errors:
        raise ValueError("Input validation failed:\n- " + "\n- ".join(schema_errors))

    genotype_markers = genotype.drop(columns="clone_id") # Exclude the clone_id column to focus on marker data.
    # Only clones present in both tables can contribute to model fitting.
    common_ids = set(phenotype["clone_id"].dropna()) & set(genotype["clone_id"].dropna())
    # Define a series of checks to validate the inputs.
    checks = [
        (phenotype.empty, "Phenotype input is empty."),
        (genotype.empty, "Genotype input is empty."),
        (
            phenotype["clone_id"].isna().any()
            or phenotype["clone_id"].astype("string").str.strip().eq("").any(),
            "Phenotype input contains missing or blank clone_id values.",
        ),
        (
            genotype["clone_id"].isna().any()
            or genotype["clone_id"].astype("string").str.strip().eq("").any(),
            "Genotype input contains missing or blank clone_id values.",
        ),
        (phenotype["clone_id"].duplicated().any(), "Phenotype input contains duplicate clone_id values."),
        (genotype["clone_id"].duplicated().any(), "Genotype input contains duplicate clone_id values."),
        (genotype_markers.empty, "Genotype input must contain at least one marker column."),
        (
            not genotype_markers.empty
            and not genotype_markers.apply(pd.api.types.is_numeric_dtype).all(),
            "All genotype marker columns must be numeric.",
        ),
        (
            not genotype_markers.empty
            and genotype_markers.apply(pd.api.types.is_numeric_dtype).all()
            and np.isinf(genotype_markers.to_numpy(dtype=float)).any(),
            "Genotype marker columns cannot contain infinite values.",
        ),
        (not common_ids, "No overlapping clone_id values were found between inputs."),
    ]
    for target in target_columns:
        if not pd.api.types.is_numeric_dtype(phenotype[target]):
            checks.append((True, f"Phenotype column {target!r} must be numeric."))
        elif np.isinf(phenotype[target].dropna().to_numpy(dtype=float)).any():
            checks.append((True, f"Phenotype column {target!r} cannot contain infinite values."))
    # Collect all failed checks and raise a single error if any are found.
    errors = [message for failed, message in checks if failed]
    if errors:
        raise ValueError("Input validation failed:\n- " + "\n- ".join(errors))
    # Count the number of samples with observed target values among the overlapping clones.
    observed_target_count = phenotype.loc[
        phenotype["clone_id"].isin(common_ids), target_columns
    ].notna().all(axis=1).sum()
    # Return a summary DataFrame with key metrics about the inputs.
    return pd.DataFrame(
        {
            "metric": [
                "phenotype clones",
                "genotype clones",
                "overlapping clones",
                "clones with observed target",
                "SNP markers",
            ],
            "value": [
                len(phenotype),
                len(genotype),
                len(common_ids),
                observed_target_count,
                genotype_markers.shape[1],
            ],
        }
    )

# Determine the number of markers to select based on the number of samples and features.
def _choose_marker_count(n_samples: int, n_features: int) -> int:
    """Pick a marker count that keeps at least MIN_SAMPLES_PER_MARKER samples per feature."""
    # Also cap by the configured maximum and by the markers actually available.
    ratio_capped = max(1, n_samples // MIN_SAMPLES_PER_MARKER)
    if n_features < 1:
        raise ValueError("At least one marker column is required.")
    return min(MAX_SELECTED_MARKERS, n_features, ratio_capped)

# Build a machine learning pipeline for the specified model family.
def _build_model_pipeline(
    n_samples: int,
    n_features: int,
    model_family: str | None = None,
    seed: int = DEFAULT_SEED,
) -> Pipeline:
    """Build one of the supported genomic prediction model pipelines."""
    selected_model_family = model_family or MODEL_FAMILY
    # Keep preprocessing in the pipeline so it is learned from each training
    # fold only, rather than leaking validation-fold information into training.
    if selected_model_family == "ridge":
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 5, 30))), # Fit a Ridge regression model with cross-validated alpha values
            ]
        )

    if selected_model_family == "elastic_net":
        # Pre-select markers so coordinate descent runs on a tractable, less
        # collinear feature set; without this, ElasticNetCV on all raw SNPs
        # (features >> samples) can stall for a very long time or fail to
        # converge before hitting max_iter.
        n_markers = _choose_marker_count(n_samples, n_features)
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("select", SelectKBest(score_func=f_regression, k=n_markers)), # Select the top k markers based on univariate regression tests
                ("scale", StandardScaler()), # Standardize features by removing the mean and scaling to unit variance
                (
                    "elastic_net",
                    ElasticNetCV( # Fit an Elastic Net model with cross-validated hyperparameters
                        l1_ratio=[0.1, 0.5, 0.9], # Mix of L1 and L2 regularization ratios to try
                        alphas=50, # Number of alpha values to try
                        cv=KFold(n_splits=3, shuffle=True, random_state=seed),
                        tol=1e-3, # Tolerance for optimization
                        max_iter=10_000, # Maximum number of iterations
                        selection="random", # Random coordinate descent selection
                        random_state=seed, # Seed for reproducibility
                        n_jobs=-1, # Use all available CPU cores
                    ),
                ),
            ]
        )
# If the selected model family is not recognized, raise an error.
    if selected_model_family != "mlp_baseline":
        raise ValueError(
            f"Unknown model family {selected_model_family!r}; choose from {MODEL_FAMILIES}."
        )

    # With only a few dozen labeled samples and 500 raw markers, the MLP needs
    # much more aggressive dimensionality reduction and regularization than the
    # linear models: pre-select markers (drops pure noise before PCA even
    # sees it), cap components in the single digits, use a tiny single hidden
    # layer, strong L2 (alpha), and 'lbfgs' - which converges far better than
    # 'adam'/'sgd' on small-n batches and needs no batch_size/early_stopping.
    # The target is standardized for fitting and predictions are mapped back to
    # the original units. Without this, a fixed alpha acts very differently on
    # traits whose scales differ by orders of magnitude, and lbfgs fails to
    # converge on large-mean traits. The scaler is fitted on each training fold.
    n_markers = _choose_marker_count(n_samples, n_features) # Determine the number of markers to select based on the sample and feature count
    n_components = min(8, max(1, n_samples // 2), n_markers)
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("select", SelectKBest(score_func=f_regression, k=n_markers)), # Select the top k markers based on univariate regression tests
            ("scale", StandardScaler()), # Standardizes selected markers to zero mean (mu = 0) and unit variance (sigma = 1) within the training fold
            (
                "pca", # Principal Component Analysis for dimensionality reduction
                PCA(
                    n_components=n_components, 
                    svd_solver="randomized", # Use randomized Singular Value Decomposition (SVD) algorithm for faster computation on large datasets
                    random_state=seed,
                ),
            ),
            (
                "mlp",
                TransformedTargetRegressor(
                    regressor=MLPRegressor(
                        hidden_layer_sizes=(8,), # Single hidden layer with 8 neurons
                        activation="relu", # ReLU activation function for non-linear transformations
                        solver="lbfgs", # Use the L-BFGS optimizer for small datasets
                        alpha=1.0, # L2 regularization strength (on the standardized target)
                        max_iter=2000, # Maximum number of iterations for the optimizer
                        random_state=seed,
                    ),
                    transformer=StandardScaler(), # Standardize y within each training fold
                ),
            ),
        ]
    )

# Fit and evaluate a single-target model using cross-validation, handling cases where fitting is not feasible.
def _fit_and_evaluate_target(
    target_name: str,
    labeled_features: pd.DataFrame,
    labeled_target: pd.Series,
    model_family: str | None = None,
    seed: int = DEFAULT_SEED,
    n_cv_folds: int | None = None,
) -> dict:
    """Cross-validate and fit a single-target model, or report why it was skipped."""
    non_null_mask = labeled_target.notna() # Mask to identify non-missing target values
    x = labeled_features.loc[non_null_mask] # Features corresponding to non-missing target values
    y = labeled_target.loc[non_null_mask] # Non-missing target values
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError(f"Target {target_name!r} must be numeric.")
    if np.isinf(y.to_numpy(dtype=float)).any():
        raise ValueError(f"Target {target_name!r} cannot contain infinite values.")
    n_samples = len(y) # Number of labeled samples for this target
    n_features = x.shape[1] # Number of available markers/features
    if n_features < 1:
        raise ValueError("At least one marker column is required.")
    # Initialize the result dictionary with basic information about the target.
    result: dict = {
        "target": target_name,
        "n_labeled_samples": n_samples,
        "n_markers_available": n_features,
        "seed": seed,
    }
    # Skip fitting if there are too few labeled samples or zero variance in the target.
    if n_samples < MIN_SAMPLES_PER_TARGET:
        result["skipped"] = True
        result["skip_reason"] = (
            f"Only {n_samples} labeled clones have {target_name!r}; "
            f"need at least {MIN_SAMPLES_PER_TARGET} to fit a model without severe overfitting risk."
        )
        return result
    # Skip fitting if the target has zero variance across labeled samples.
    if y.nunique() <= 1:
        result["skipped"] = True
        result["skip_reason"] = f"{target_name!r} has zero variance across labeled clones; nothing to predict."
        return result

    if n_cv_folds is None:
        # Use fewer folds for smaller labeled sets, within configured limits.
        n_folds = max(MIN_CV_FOLDS, min(MAX_CV_FOLDS, n_samples // 5))
        n_folds = min(n_folds, n_samples)
    else:
        if not MIN_CV_FOLDS <= n_cv_folds <= n_samples:
            raise ValueError(
                f"n_cv_folds must be between {MIN_CV_FOLDS} and {n_samples}; "
                f"got {n_cv_folds}."
            )
        n_folds = n_cv_folds
    selected_model_family = model_family or MODEL_FAMILY # Choose the model family, defaulting to the configured global option if not specified
    # CV uses fold-safe dimensions; the independent final refit uses all labels.
    min_train_fold_size = n_samples - math.ceil(n_samples / n_folds)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_model = _build_model_pipeline(
        min_train_fold_size, n_features, selected_model_family, seed=seed
    )
    oof_predictions = cross_val_predict(cv_model, x, y, cv=cv)
    model = _build_model_pipeline(
        n_samples, n_features, selected_model_family, seed=seed
    )
    model.fit(x, y)
    n_selected_markers = (
        int(model.named_steps["select"].get_support().sum())
        if "select" in model.named_steps
        else n_features
    )
    n_cv_selected_markers = (
        cv_model.named_steps["select"].k
        if "select" in cv_model.named_steps
        else n_features
    )
    # Compute the number of samples per selected marker to assess overfitting risk.
    samples_per_marker = round(n_samples / n_selected_markers, 2) # Calculate the number of samples per selected marker to evaluate overfitting risk
    if selected_model_family == "ridge":
        overfitting_risk = "controlled_by_ridge_shrinkage"
    elif selected_model_family.startswith("mlp_"):
        overfitting_risk = "controlled_by_dimension_reduction_and_regularization"
    elif samples_per_marker < MIN_SAMPLES_PER_MARKER:
        overfitting_risk = "high"
    elif samples_per_marker < 2 * MIN_SAMPLES_PER_MARKER:
        overfitting_risk = "moderate"
    else:
        overfitting_risk = "low"
    # Update the result dictionary with model evaluation metrics and other relevant information.
    result.update(
        {
            "model_family": selected_model_family,
            "skipped": False,
            "skip_reason": None,
            "n_cv_folds": n_folds,
            "n_selected_markers": n_selected_markers,
            "n_cv_selected_markers": n_cv_selected_markers,
            "n_pca_components": (
                int(model.named_steps["pca"].n_components_)
                if "pca" in model.named_steps else None
            ),
            "n_cv_pca_components": (
                cv_model.named_steps["pca"].n_components
                if "pca" in cv_model.named_steps else None
            ),
            "samples_per_selected_marker": samples_per_marker,
            "overfitting_risk": overfitting_risk,
            "cv_mae": float(mean_absolute_error(y, oof_predictions)),
            "cv_rmse": float(mean_squared_error(y, oof_predictions) ** 0.5),
            "cv_r2": float(r2_score(y, oof_predictions)),
        }
    )

    result["_fitted_model"] = model # Store the final fitted model in the result dictionary
    result["_oof_predictions"] = pd.Series(oof_predictions, index=x.index) # Store the out-of-fold predictions in the result dictionary
    return result # Return the result dictionary containing model evaluation metrics and other relevant information

# Compare different model families using identical out-of-fold splits and evaluation metrics.
def _compare_model_families_loaded(
    phenotype: pd.DataFrame,
    genotype: pd.DataFrame,
    target: str,
    model_families: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict]]:
    """Compare model families on already-loaded inputs.

    Returns the comparison table and the full outcome of every fit, keyed by
    (target, model_family), so callers can reuse the fitted models and
    out-of-fold predictions instead of refitting them.
    """
    source_column = TARGET_SOURCE_COLUMNS.get(target, target) # Determine the source column for the target in the phenotype data
    if target not in TARGET_COLUMNS or source_column not in phenotype.columns:
        raise ValueError(f"Target {target!r} is not supported by the phenotype input.")
    validate_inputs(phenotype, genotype, targets=[source_column])
    invalid_models = sorted(set(model_families) - set(MODEL_FAMILIES)) # Identify any model families that are not supported
    if invalid_models:
        raise ValueError(f"Unknown model families {invalid_models}; choose from {MODEL_FAMILIES}.")
    # Identify marker columns by excluding the "clone_id" column from the genotype data.
    marker_columns = [column for column in genotype.columns if column != "clone_id"]
    labeled_df = phenotype.merge(genotype, on="clone_id", how="inner").set_index("clone_id")
    results = []
    outcomes: dict[tuple[str, str], dict] = {}
    for model_family in model_families:
        outcome = _fit_and_evaluate_target(
            target,
            labeled_df[marker_columns],
            labeled_df[source_column],
            model_family=model_family,
            seed=seed,
        )
        outcomes[(target, model_family)] = outcome
        results.append(
            {
                "model_family": model_family,
                "target": target,
                "seed": outcome.get("seed"),
                "n_labeled_samples": outcome["n_labeled_samples"],
                "n_markers": outcome["n_markers_available"],
                "n_selected_markers": outcome.get("n_selected_markers"),
                "n_cv_selected_markers": outcome.get("n_cv_selected_markers"),
                "n_pca_components": outcome.get("n_pca_components"),
                "n_cv_pca_components": outcome.get("n_cv_pca_components"),
                "cv_folds": outcome.get("n_cv_folds"),
                "cv_r2": outcome.get("cv_r2"),
                "cv_rmse": outcome.get("cv_rmse"),
                "cv_mae": outcome.get("cv_mae"),
                "overfitting_risk": outcome.get("overfitting_risk"),
            }
        )

    comparison = pd.DataFrame(results).sort_values("cv_r2", ascending=False).reset_index(drop=True)
    return comparison, outcomes

# Function to compare model families across multiple targets
def compare_model_families_across_targets(
    phenotype_path: str | Path,
    genotype_path: str | Path,
    output_dir: str | Path,
    targets: list[str] | None = None,
    model_families: tuple[str, ...] = MODEL_FAMILIES,
    seed: int = DEFAULT_SEED,
    return_outcomes: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[tuple[str, str], dict]]:
    """Compare model families across targets and save one consolidated results file.

    With return_outcomes=True, also returns every fit's full outcome (fitted
    model, out-of-fold predictions, metrics) keyed by (target, model_family).
    Pass that dict as `outcomes=` to run_capstone_analysis and
    top_candidates_by_model to reuse these fits instead of refitting.
    """
    # Load the inputs once and reuse them for every target.
    phenotype, genotype = load_inputs(phenotype_path, genotype_path)
    target_columns = targets if targets is not None else TARGET_COLUMNS
    comparison_frames = []
    all_outcomes: dict[tuple[str, str], dict] = {}
    for target in target_columns:
        try:
            comparison, outcomes = _compare_model_families_loaded(
                phenotype,
                genotype,
                target=target,
                model_families=model_families,
                seed=seed,
            )
        except ValueError as exc:
            print(f"{target}: skipped ({exc})")
            continue
        comparison_frames.append(comparison)
        all_outcomes.update(outcomes)

    if not comparison_frames:
        raise ValueError("No supported targets could be compared.")

    comparison = pd.concat(comparison_frames, ignore_index=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path / "model_comparison_all_targets.csv", index=False) # Save the consolidated comparison results for all targets
    if return_outcomes:
        return comparison, all_outcomes
    return comparison


def _reusable_outcome(
    outcomes: dict[tuple[str, str], dict] | None,
    target: str,
    model_family: str,
    seed: int,
) -> dict | None:
    """Return a copy of a stored outcome fitted with the same seed, or None.

    Outcomes come from compare_model_families_across_targets(return_outcomes=True),
    which uses the default CV fold count, so a reused fit matches a fresh one.
    A shallow copy lets callers pop keys without altering the shared dict.
    """
    if not outcomes:
        return None
    outcome = outcomes.get((target, model_family))
    if outcome is None or outcome.get("seed") != seed:
        return None
    return dict(outcome)

# Function to get top candidates by model family
def top_candidates_by_model(
    phenotype_path: str | Path,
    genotype_path: str | Path,
    target: str = TARGET_COLUMN,
    model_families: tuple[str, ...] = MODEL_FAMILIES,
    top_n: int = 5,
    seed: int = DEFAULT_SEED,
    outcomes: dict[tuple[str, str], dict] | None = None,
) -> pd.DataFrame:
    """Return the top unphenotyped candidates for one target, ranked separately by each model family.

    Pass `outcomes` from compare_model_families_across_targets(return_outcomes=True)
    to reuse those fitted models; any model not found there (or fitted with a
    different seed) is fitted here as before.
    """
    phenotype, genotype = load_inputs(phenotype_path, genotype_path)
    source_column = TARGET_SOURCE_COLUMNS.get(target, target)
    if target not in TARGET_COLUMNS or source_column not in phenotype.columns:
        raise ValueError(f"Target {target!r} is not supported by the phenotype input.")
    validate_inputs(phenotype, genotype, targets=[source_column])
    invalid_models = sorted(set(model_families) - set(MODEL_FAMILIES))
    if invalid_models:
        raise ValueError(f"Unknown model families {invalid_models}; choose from {MODEL_FAMILIES}.")
    # Merge phenotype and genotype data to identify labeled and novel candidates
    marker_columns = [column for column in genotype.columns if column != "clone_id"]
    all_candidate_features = genotype.set_index("clone_id")[marker_columns] 
    labeled_df = phenotype.merge(genotype, on="clone_id", how="inner").set_index("clone_id")
    # A clone with any phenotype record is not an unlabeled breeding candidate,
    # even if its value for this particular target is missing.
    novel_features = all_candidate_features.loc[~all_candidate_features.index.isin(labeled_df.index)]

    frames = []
    for model_family in model_families:
        outcome = _reusable_outcome(outcomes, target, model_family, seed)
        if outcome is None:
            outcome = _fit_and_evaluate_target(
                target,
                labeled_df[marker_columns],
                labeled_df[source_column],
                model_family=model_family,
                seed=seed,
            )
        if outcome["skipped"]:
            raise ValueError(outcome["skip_reason"])
        predictions = pd.Series(outcome["_fitted_model"].predict(novel_features), index=novel_features.index)
        top = predictions.sort_values(ascending=False).head(top_n)
        frames.append(
            pd.DataFrame(
                {
                    "model_family": model_family,
                    "rank": range(1, len(top) + 1), 
                    "clone_id": top.index,
                    "predicted_value": top.values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)

# Function to run the full capstone analysis pipeline
def run_capstone_analysis(
    phenotype_path: str | Path,
    genotype_path: str | Path,
    output_dir: str | Path,
    targets: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    outcomes: dict[tuple[str, str], dict] | None = None,
) -> dict:
    """Train on labeled clones and predict each target trait for every genotyped clone.

    Pass `outcomes` from compare_model_families_across_targets(return_outcomes=True)
    to reuse its primary-model (Elastic Net) fits; any target not found there
    (or fitted with a different seed) is fitted here as before.
    """
    phenotype, genotype = load_inputs(phenotype_path, genotype_path)
    target_columns = targets if targets is not None else [
        target for target in TARGET_COLUMNS
        if TARGET_SOURCE_COLUMNS[target] in phenotype.columns
    ]
    unsupported_targets = sorted(set(target_columns) - set(TARGET_COLUMNS))
    if unsupported_targets:
        raise ValueError(f"Unsupported targets: {', '.join(unsupported_targets)}")
    alignment = validate_inputs(
        phenotype, genotype,
        targets=[TARGET_SOURCE_COLUMNS[target] for target in target_columns],
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_columns = [column for column in genotype.columns if column != "clone_id"]
    all_candidate_features = genotype.set_index("clone_id")[marker_columns]

    # The inner join restricts training to clones with both phenotype and
    # genotype records; predictions below are made for every genotyped clone.
    labeled_df = phenotype.merge(genotype, on="clone_id", how="inner").set_index("clone_id")
    if len(labeled_df) < MIN_SAMPLES_PER_TARGET:
        raise ValueError(
            f"Only {len(labeled_df)} clones have both phenotype and genotype data; "
            f"need at least {MIN_SAMPLES_PER_TARGET} to fit any target model."
        )
    labeled_features = labeled_df[marker_columns]
    # Initialize structures to store results and combined predictions
    target_results: dict[str, dict] = {}
    combined_predictions = pd.DataFrame(index=all_candidate_features.index)
    combined_predictions["is_training_clone"] = combined_predictions.index.isin(labeled_df.index) # Mark clones that are part of the training set
    # Loop over each target trait and fit models, generate predictions, and rank candidates
    for target_name in target_columns:
        source_column = TARGET_SOURCE_COLUMNS.get(target_name, target_name)
        if source_column not in labeled_df.columns: # Skip targets for which the source column is not available in the labeled data
            continue
        outcome = _reusable_outcome(outcomes, target_name, MODEL_FAMILY, seed)
        if outcome is None:
            outcome = _fit_and_evaluate_target(
                target_name, labeled_features, labeled_df[source_column], seed=seed
            )
        if outcome["skipped"]:
            target_results[target_name] = outcome
            continue

        model = outcome.pop("_fitted_model") # Extract the fitted model from the outcome dictionary
        oof_predictions = outcome.pop("_oof_predictions") # Extract out-of-fold predictions for training clones
        # Extract candidate predictions for all clones using the fitted model
        candidate_predictions = pd.Series(
            model.predict(all_candidate_features), index=all_candidate_features.index # Predict outcomes for all candidate clones
        )
        # Create a DataFrame to store ranked predictions for the current target
        ranked = pd.DataFrame(
            {
                "clone_id": all_candidate_features.index,
                f"predicted_{target_name}": candidate_predictions.values,
            }
        )
        ranked[f"observed_{target_name}"] = ranked["clone_id"].map(labeled_df[source_column]) # Map observed values for the target
        ranked["is_training_clone"] = ranked["clone_id"].isin(oof_predictions.index) # Mark training clones
        # Cross-validated (out-of-fold) predictions for training clones, for an honest
        # observed-vs-predicted diagnostic; NaN for candidates outside the labeled set.
        ranked[f"cv_predicted_{target_name}"] = ranked["clone_id"].map(oof_predictions)
        ranked = ranked.sort_values(f"predicted_{target_name}", ascending=False).reset_index(drop=True) # Sort candidates by predicted values

        combined_predictions[f"predicted_{target_name}"] = candidate_predictions # Store predicted values for all candidates
        combined_predictions[f"observed_{target_name}"] = combined_predictions.index.map(
            labeled_df[source_column] # Map observed values for all candidates
        )
        combined_predictions[f"cv_predicted_{target_name}"] = oof_predictions.reindex(
            combined_predictions.index # Align out-of-fold predictions with all candidates
        )
        # Store the ranked clones in the outcome dictionary for the current target
        outcome["ranked_clones"] = ranked # Store the ranked clones for the current target
        target_results[target_name] = outcome # Store the outcome, including ranked clones, for the current target
    # After processing all targets, reset the index of combined predictions and save to CSV
    combined_predictions = combined_predictions.reset_index().rename(columns={"index": "clone_id"})
    combined_predictions.to_csv(output_dir / "ranked_clones_multi_index.csv", index=False)
    # Save the combined ranked clones to CSV for all targets
    metrics_summary = {
        name: {k: v for k, v in result.items() if k != "ranked_clones"}
        for name, result in target_results.items()
    }

    return {
        "alignment": alignment,
        "targets": target_results,
        "combined_ranked_clones": combined_predictions,
        "metrics": metrics_summary,
    }

# Parse command-line arguments for the capstone pipeline
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenotype", type=Path, default=DEFAULT_PHENOTYPE_PATH, help="Path to clean phenotype CSV.") 
    parser.add_argument("--genotype", type=Path, default=DEFAULT_GENOTYPE_PATH, help="Path to clean genotype CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write CSV/JSON outputs.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for cross-validation and stochastic estimators (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Elastic Net with Ridge and MLP instead of running the full ranking pipeline.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target to compare in --compare mode (default: compare all targets).",
    )
    parser.add_argument(
        "--model-families",
        default="elastic_net,ridge,mlp_baseline",
        help=(
            "Comma-separated model families to compare in --compare mode, from "
            f"{MODEL_FAMILIES} (default: 'elastic_net,ridge,mlp_baseline')."
        ),
    )
    return parser.parse_args(argv)

# Main entry point for the capstone pipeline
def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.compare:
        model_families = tuple(name.strip() for name in args.model_families.split(","))
        targets = [args.target] if args.target else None
        comparison = compare_model_families_across_targets(
            args.phenotype,
            args.genotype,
            args.output_dir,
            targets=targets,
            model_families=model_families,
            seed=args.seed,
        )
        print(comparison.to_string(index=False))
        print(f"\nComparison written to: {args.output_dir / 'model_comparison_all_targets.csv'}")
        return

    result = run_capstone_analysis(args.phenotype, args.genotype, args.output_dir, seed=args.seed)
    for target_name, outcome in result["targets"].items():
        if outcome.get("skipped"):
            print(f"{target_name}: skipped ({outcome['skip_reason']})")
        else:
            print(f"{target_name}: cv_r2={outcome['cv_r2']:.3f}  cv_rmse={outcome['cv_rmse']:.4f}")
    print(f"\nRanked clones written to: {args.output_dir}")
    print("Use --compare to generate model-comparison metrics separately.")


if __name__ == "__main__":
    main()
