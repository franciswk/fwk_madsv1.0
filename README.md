# Sweetpotato Genomic Prediction

This project uses SNP markers to predict five sweetpotato indices and rank
genotyped clones. It compares Elastic Net, Ridge, and a neural network baseline.
Phenotypes are partially synthetic, and genotypes are simulated; results do not
establish real-world predictive accuracy.

## Project files

- **Dependencies:** [requirements.txt](requirements.txt).
- **Required sweetpotato inputs:** `data/raw/raw_phenotypes.csv` (54 clones;
  literature-based and synthetic records, curated by hand) and
  `data/raw/raw_genotypes.csv` (simulated from the phenotypes; see
  [Regenerating the genotypes](#regenerating-the-genotypes)).
- **Simulated signal:** the genotype simulator shifts the allele frequencies of
  25 random markers per index in proportion to each labeled clone's index
  value. The 10,000 unlabeled candidates get no shift, so cross-validated
  accuracy measures recovery of that planted signal, and candidate rankings
  cannot be validated. The current `raw_genotypes.csv` embeds the revised indices.
- **Notebooks:**
  - [notebooks/00_generate_raw_data.ipynb](notebooks/00_generate_raw_data.ipynb)
    simulates genotypes from the phenotype file (optional).
  - [notebooks/02_genomic_prediction_analysis.ipynb](notebooks/02_genomic_prediction_analysis.ipynb)
    runs the sweetpotato analysis.
  - [notebooks/03_seed_fold_sensitivity.ipynb](notebooks/03_seed_fold_sensitivity.ipynb)
    evaluates sensitivity to random seeds and outer cross-validation fold counts.
  - [notebooks/01_prepare_external_benchmarks.ipynb](notebooks/01_prepare_external_benchmarks.ipynb)
    prepares external benchmark data.
  - [notebooks/02_external_data_validation.ipynb](notebooks/02_external_data_validation.ipynb)
    runs external validation.
- **Dashboard:** [app.py](app.py) is a Streamlit app that runs the pipeline and
  shows the results (see [Dashboard](#dashboard)).
- **Python scripts:**
  - [src/etl_pipeline.py](src/etl_pipeline.py) cleans the sweetpotato inputs.
  - [src/scoring_pipeline.py](src/scoring_pipeline.py) calculates breeding indices.
  - [src/capstone_pipeline.py](src/capstone_pipeline.py) trains and evaluates models.
  - [src/data_generation/simulate_realistic_genotypes.py](src/data_generation/simulate_realistic_genotypes.py)
    simulates the genotype file from the phenotypes.
  - [src/external_validation/prepare_ag2p_benchmark.py](src/external_validation/prepare_ag2p_benchmark.py)
    converts an external benchmark dataset.
- **Optional external benchmark inputs:** for external validation, provide
  `curated_geno_pheno.rdata` for each crop under
  `data/benchmarks/learnable/raw/<crop>/`, using `commonbean`, `durum_wheat`,
  `maize`, `rice210`, and `soybean` as the crop names.

## Getting started

From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sweetpotato-capstone --display-name "Python (Sweetpotato Capstone)"
```

Open [the genomic prediction notebook](notebooks/02_genomic_prediction_analysis.ipynb),
select the **Python** kernel, and choose how much of the notebook to run.
**Run All** executes the setup and ETL cells, primary-model evaluation and
ranking, model-family comparisons, timing benchmark, and candidate-ranking
cells. 
To run only one analysis, first run the setup cells in order, then run
the cell for that analysis (and the ETL cell first if the processed inputs need
to be refreshed). 
The model-comparison cell and candidate-ranking cell can be
run independently after setup.

The inputs are `data/raw/raw_phenotypes.csv` and `data/raw/raw_genotypes.csv`.
Keep these raw files: the notebook does not regenerate them.

## Dashboard

From the project root, with the environment active:

```powershell
streamlit run app.py
```

- **Sidebar:** **Run analysis** runs the ETL, the model comparison and the rankings
  (about 30 s); **Run sensitivity** runs 3 seeds × 3 fold counts (about 1 min).
  Both overwrite `data/processed/` and the CSVs in `outputs/`, exactly as the
  notebooks do.
- **Tabs:** data and index distributions, model performance, top-N candidates
  per index and model (downloadable), seed and fold sensitivity, and external
  benchmarks.
- With no results yet, click **Run analysis** first. External benchmarks come
  from notebook 02_external_data_validation.

## How the pipeline works

The notebook calls these scripts in sequence:

1. **Clean the data:** [etl_pipeline.py](src/etl_pipeline.py) normalizes clone
   IDs, converts phenotype and SNP values to numeric form, and resolves
   duplicates. Missing SNP calls are retained for model-time imputation.
   Missing or blank clone IDs are rejected. Phenotype duplicates keep the most
   measured row, breaking ties by original row order
2. **Calculate breeding indices:** [scoring_pipeline.py](src/scoring_pipeline.py),
   called by ETL, derives disease resistance (DRI), eating quality (EQI), and
   agronomic yield (AYI) scores from available traits. Market Index is their
   equal-weight mean; Parental Value weights DRI and AYI at 40% each and EQI at
   20%. DRI averages one score per disease, preferring the measured value over
   the literature rating. A clone with no disease, or no yield and maturity,
   data gets a missing index rather than 0 and is dropped by ETL. These are
   proxy scores, with fallback rules for missing traits.
3. **Train and evaluate:** [capstone_pipeline.py](src/capstone_pipeline.py)
   joins clones with both phenotype and genotype data and trains a separate
   model for each index. Elastic Net is the primary model; Ridge and a small
   neural network provide comparisons. Shuffled cross-validation produces
   out-of-fold predictions and R², RMSE, and MAE scores. Imputation and other
   preprocessing are fitted within each outer training fold. Elastic Net tunes
   its penalties with seeded, shuffled three-fold splits of that preprocessed
   training data; preprocessing is not independently refit within those inner
   splits. Entirely missing marker columns are retained and imputed to zero.
4. **Predict and rank:** The primary models are refit on the labeled clones and
   predict indices for all genotyped clones. The notebook also compares the
   top unphenotyped candidates across model families.

With 54 labeled clones and five outer folds, Elastic Net and MLP select 14
markers for CV and 18 for the final refit. MLP then reduces these to at most
eight PCA components, and fits on a standardized target (scaled within each
training fold, with predictions mapped back to the original units) so its
fixed L2 penalty behaves the same across traits of very different scales. Results distinguish `n_cv_selected_markers` from
`n_selected_markers` (the fitted final selector count); the samples-per-marker
diagnostic uses the final count. Ridge retains all markers.

The notebook runs each model five times with the same random seed and
cross-validation splits to measure runtime, not seed or fold-count sensitivity.

To run the scripts without the notebook:

```powershell
python src\etl_pipeline.py
python src\capstone_pipeline.py
python src\capstone_pipeline.py --compare
```

## Regenerating the genotypes

Optional. Input: `data/raw/raw_phenotypes.csv` only.
[src/data_generation/simulate_realistic_genotypes.py](src/data_generation/simulate_realistic_genotypes.py)
scores it with the project ETL (current index formulas), generates 9,946 seeded
candidate IDs, and simulates 500 SNPs (allele-frequency spectrum, six
breeding-program subgroups, 10-marker LD blocks, planted signal, messiness).

- **Notebook (recommended):** [notebooks/00_generate_raw_data.ipynb](notebooks/00_generate_raw_data.ipynb)
  writes to `data/generated/<run>/` (git-ignored), checks the ETL join and
  cross-validated R², and copies the file into `data/raw/` only if
  `PROMOTE_GENOTYPES_TO_RAW = True`.
- **Script:** `python -m src.data_generation.simulate_realistic_genotypes`
  overwrites `data/raw/raw_genotypes.csv` after backing it up to
  `data/raw/backup_<timestamp>/`.

After replacing the genotypes, re-run the ETL and notebooks 02 and 03.

## Outputs

- `data/processed/`: cleaned phenotype and genotype CSVs and `etl_report.json`.
  Not tracked in git; created by the ETL step (notebook 02 or
  `python src\etl_pipeline.py`).
- `outputs/ranked_clones_multi_index.csv`: predictions, observed values, and
  cross-validated predictions for the modeled indices.
- `outputs/model_comparison_all_targets.csv`: cross-validation results for the
  model families.
- `outputs/top10_candidates_by_model_<target>.csv`: the notebook's top-candidate
  comparison. Set `RANKING_TARGET` in the notebook to choose the target.

## External benchmarks

The separate external-validation workflow uses five AG2P-DISC datasets:
[common bean](https://www.kaggle.com/datasets/ag2p-disc/00041-agronomictraitscommonbean),
[durum wheat](https://www.kaggle.com/datasets/ag2p-disc/00026-globaldurumwheatpanel),
[maize](https://www.kaggle.com/datasets/ag2p-disc/00030-multigenerationgenomicpredictionmaize),
[rice](https://www.kaggle.com/datasets/ag2p-disc/00080-targetorientedprioritizationrice210),
and [soybean](https://www.kaggle.com/datasets/ag2p-disc/00023-genomicpredictionmodeltraitssoybean).

Download and extract each dataset's `curated_geno_pheno.rdata` into
`data/benchmarks/learnable/raw/<crop>/`, using the crop names `commonbean`,
`durum_wheat`, `maize`, `rice210`, and `soybean`. Then run
[the preparation notebook](notebooks/01_prepare_external_benchmarks.ipynb),
followed by [the validation notebook](notebooks/02_external_data_validation.ipynb).
Benchmark results are written under `outputs/benchmarks/learnable/` and do not
establish predictive accuracy for sweetpotato.

The benchmark preserves the sweetpotato pipeline caps: Elastic Net and MLP
select at most 50 markers, MLP uses at most eight PCA components, and Ridge
uses all available markers. Results compare these configured pipelines, not
equally optimized model families. Per-trait `n_samples` counts only overlapping
clones with an observed target; maize has 3,527 samples for traits 1–3 and
3,526 for traits 4–6, rather than all 3,528 genotype rows.

## Repository and licensing

The project's source code, notebooks, and user guide are available in the
[fwk_capstone_v1.0 GitHub repository](https://github.com/franciswk/fwk_capstone_v1.0)
[34]. Clone or download the repository and follow its README for setup and
analysis instructions; dependencies are listed in
[requirements.txt](requirements.txt). The README describes the main workflow
and generated output locations. The source code is licensed under the
[MIT License](LICENSE). Original project documentation and figures, distributed
separately from the source code, are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
