# Sweetpotato Genomic Prediction

This project uses SNP markers to predict five sweetpotato indices and rank
genotyped clones. It compares Elastic Net, Ridge, and a neural network baseline.
Phenotypes are partially synthetic, and genotypes are simulated; results do not
establish real-world predictive accuracy.

## Getting started

From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sweetpotato-capstone --display-name "Python (Sweetpotato Capstone)"
```

Open [the genomic prediction notebook](notebooks/02_genomic_prediction_analysis.ipynb),
select **Python** kernel, and run all cells. The notebook cleans
the raw data, trains and compares models, and displays interactive plots.

The inputs are `data/raw/raw_phenotypes.csv` and `data/raw/raw_genotypes.csv`.
Keep these raw files: the notebook does not regenerate them.

## How the pipeline works

The notebook calls these scripts in sequence:

1. **Clean the data:** [etl_pipeline.py](src/etl_pipeline.py) normalizes clone
   IDs, converts phenotype and SNP values to numeric form, and resolves
   duplicates. Missing SNP calls are retained for model-time imputation.
2. **Calculate breeding indices:** [scoring_pipeline.py](src/scoring_pipeline.py),
   called by ETL, derives disease resistance (DRI), eating quality (EQI), and
   agronomic yield (AYI) scores from available traits. Market Index is their
   equal-weight mean; Parental Value weights DRI and AYI at 40% each and EQI at
   20%. These are proxy scores, with fallback rules for missing traits.
3. **Train and evaluate:** [capstone_pipeline.py](src/capstone_pipeline.py)
   joins clones with both phenotype and genotype data and trains a separate
   model for each index. Elastic Net is the primary model; Ridge and a small
   neural network provide comparisons. Shuffled cross-validation produces
   out-of-fold predictions and R², RMSE, and MAE scores. Imputation and other
   preprocessing are fitted within each training fold.
4. **Predict and rank:** The primary models are refit on the labeled clones and
   predict indices for all genotyped clones. The notebook also compares the
   top unphenotyped candidates across model families.

The notebook runs each model five times with the same random seed and
cross-validation splits to measure runtime.

## Outputs

- `data/processed/`: cleaned phenotype and genotype CSVs and `etl_report.json`.
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
