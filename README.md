# Sweetpotato Genomic Prediction Capstone

This is a notebook-first genomic prediction pipeline: predict sweetpotato breeding
values (market index, disease resistance, and eating quality) from SNP
genotype markers, train on clones with known phenotypes (real commercial cultivars
plus synthetic breeding lines), and rank unphenotyped breeding candidates.

It has no dependency on the Streamlit dashboard or its source code.

## Quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sweetpotato-capstone --display-name "Python (Sweetpotato Capstone)"
```

Run the pipeline end to end:

```powershell
python src\etl_pipeline.py     # clean the raw messy CSVs -> data/processed/
python src\run_smoke_tests.py  # sanity-check every script before trusting results
python src\capstone_pipeline.py                                  # rank every genotyped clone -> outputs/
python src\capstone_pipeline.py --compare --target market_index  # Elastic Net vs Ridge/MLP comparison -> outputs/
```

`capstone_pipeline.py` defaults to reading `data/processed/phenotype_clean.csv` +
`data/processed/genotype_clean.csv` and writing to `outputs/`; pass `--phenotype`,
`--genotype`, or `--output-dir` to override any of those. Run
`python src\capstone_pipeline.py --help` to see all flags (including `--target` and
`--model-families` for `--compare` mode).

Then open [the manuscript companion notebook](../notebooks/02_genomic_prediction_analysis.ipynb), select the
**Python (Sweetpotato Capstone)** kernel, and run all cells. The notebook runs
the ETL stage itself, so the manual `etl_pipeline.py`/`capstone_pipeline.py` steps
above are optional - they're there so you can run/inspect each stage from the
command line without opening Jupyter.

The companion follows [the manuscript](capstone_francis.txt): Methodology, then
Results and Discussion (data structure, Elastic Net prediction power, model
comparison, computational checks, explainability, index associations, rankings,
and overfitting controls), followed by interpretation and saved artifacts.
Its reading guide maps manuscript captions to the existing figure filenames.
Set `RANKING_TARGET` in setup to change the candidate-ranking objective; the
Market Index diagnostic has a separate `DIAGNOSTIC_TARGET`.

The fixed-configuration timing cell does not reproduce the manuscript's full
seed/fold sensitivity experiment. Public-crop validation is a separate workflow,
not executed by this sweetpotato notebook. Both coverage limits are marked
explicitly rather than presenting the local results as those experiments.

## External benchmark data

The external-validation notebook uses five AG2P-DISC datasets:

| Dataset slug | Kaggle source |
|---|---|
| `commonbean` | [Agronomic Traits Common Bean](https://www.kaggle.com/datasets/ag2p-disc/00041-agronomictraitscommonbean) |
| `durum_wheat` | [Global Durum Wheat Panel](https://www.kaggle.com/datasets/ag2p-disc/00026-globaldurumwheatpanel) |
| `maize` | [Multi-Generation Genomic Prediction Maize](https://www.kaggle.com/datasets/ag2p-disc/00030-multigenerationgenomicpredictionmaize) |
| `rice210` | [Target-Oriented Prioritization Rice 210](https://www.kaggle.com/datasets/ag2p-disc/00080-targetorientedprioritizationrice210) |
| `soybean` | [Genomic Prediction Model Traits Soybean](https://www.kaggle.com/datasets/ag2p-disc/00023-genomicpredictionmodeltraitssoybean) |

Download and extract each dataset from its Kaggle page into
`data/benchmarks/learnable/raw/<dataset_slug>/`. Each folder should contain
`curated_geno_pheno.rdata`. Alternatively, install the Kaggle CLI and configure
authentication as described in the [Kaggle CLI documentation](https://github.com/Kaggle/kaggle-api#authentication),
then run the following in PowerShell from the `sweetpotato_capstone_notebook`
folder:

```powershell
python -m pip install kaggle
$base = ".\data\benchmarks\learnable\raw"
$datasets = @{
    commonbean  = "ag2p-disc/00041-agronomictraitscommonbean"
    durum_wheat = "ag2p-disc/00026-globaldurumwheatpanel"
    maize       = "ag2p-disc/00030-multigenerationgenomicpredictionmaize"
    rice210     = "ag2p-disc/00080-targetorientedprioritizationrice210"
    soybean     = "ag2p-disc/00023-genomicpredictionmodeltraitssoybean"
}
foreach ($crop in $datasets.Keys) {
    $dest = Join-Path $base $crop
    New-Item -ItemType Directory -Force $dest | Out-Null
    python -m kaggle datasets download -d $datasets[$crop] -p $dest --unzip
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $crop" }
}
```

Existing extracted files may prompt for replacement. Keep any existing data you
want to preserve before downloading again.

To prepare the CSVs, install the project requirements (including `pyreadr`) and
run the following from the same folder:

```powershell
python -m pip install -r docs\requirements.txt
foreach ($crop in @("commonbean", "durum_wheat", "maize", "rice210", "soybean")) {
    $source = ".\data\benchmarks\learnable\raw\$crop\curated_geno_pheno.rdata"
    python src\external_validation\prepare_ag2p_benchmark.py $source $crop
    if ($LASTEXITCODE -ne 0) { throw "Preparation failed for $crop" }
}
```

Preparation overwrites the corresponding genotype and phenotype CSVs in
`data/benchmarks/learnable/processed/`. Then run
[the external-validation notebook](../notebooks/02_external_data_validation.ipynb)
from top to bottom. Its within-crop results benchmark the workflow; they do not
establish predictive accuracy for sweetpotato.

## Pipeline architecture

```mermaid
flowchart LR
    subgraph gen["Data generation (run once / when re-seeding data)"]
        A[data/raw/pre_etl_backup\npristine originals] -->|generate_messy_raw_data.py| B[raw_phenotypes.csv\n+ raw_genotypes.csv\n(messy, has synthetic clones)]
        B -->|simulate_realistic_genotypes.py| C[raw_genotypes.csv\n(realistic SNP structure +\nembedded causal signal)]
    end
    subgraph etl["ETL (run every time raw data changes)"]
        C -->|etl_pipeline.py: extract/transform/load| D[data/processed/\nphenotype_clean.csv\ngenotype_clean.csv\netl_report.json]
    end
    subgraph model["Modeling (run every analysis)"]
        D -->|capstone_pipeline.py: run_capstone_analysis| E[outputs/\nranked_clones_multi_index.csv\nmodel_comparison_all_targets.csv]
    end
    subgraph nb["Notebook: orchestrates ETL + modeling, adds charts"]
        F[notebooks/02_genomic_prediction_analysis.ipynb]
    end
    D --> F --> E
```

Everything downstream of `data/raw/` is regenerable. If you break something,
delete `data/processed/*` and `outputs/*` and re-run the pipeline above - the
only files that must never be regenerated from scratch (they'd lose the
seeded messiness/signal) are the ones under `data/raw/`, which is why a
backup of the pristine originals lives in `data/raw/pre_etl_backup/` and
`data/raw/pre_realism_backup/`.

## `data/` - inputs, intermediate, and backups

| Path | What it is | Why it exists |
|---|---|---|
| `data/raw/raw_phenotypes.csv` | The **only** phenotype source used by the active pipeline. 40 clones: 25 real commercial sweetpotato cultivars (measured + literature trait data) plus 15 synthetic `MSU-SYN-0xx` breeding lines. Deliberately messy: mixed-case/whitespace `clone_id`, alternate missing tokens (`NA`, `TBD`, `--`, `unknown`), percent-as-string values, and a few duplicate/re-entered rows. | Keeps results traceable to real cultivars (per project decision to prioritize realism over pure synthetic data), while giving the ETL stage genuine, representative messiness to clean instead of trusting a suspiciously perfect CSV. |
| `data/raw/raw_genotypes.csv` | The **only** genotype source used by the active pipeline. 10,000 clones x 500 SNP markers (`SNP_0001`...`SNP_0500`). ~40 rows are keyed (with messy casing variants) to the phenotype clone_ids above; the rest are unlabeled synthetic breeding-line candidates (`UF_`, `CIP_`, `LSU_`, `UGA_`, `NCSU_`, `USDA_` prefixes). Simulated with realistic population genetics (see `src/simulate_realistic_genotypes.py`) plus the same kind of messiness as the phenotype file (missing calls, malformed tokens, duplicate rows). | This is what makes genomic prediction possible: a large genotyped candidate pool where only a handful of clones have known phenotypes. Realistic MAF/LD/population-structure plus embedded causal-marker signal is what gives the model something genuine to learn (see "Why the genotype data was rebuilt" below). |
| `data/raw/pre_etl_backup/` | Untouched originals of the two files above, from *before* `generate_messy_raw_data.py` ever ran (pristine 25-row phenotype table; pure-random 10,000x500 genotype matrix with zero embedded signal). | Safety net - these backups are the only way to recover the starting point if a generation script needs to be re-run from scratch. |
| `data/raw/pre_realism_backup/` | The genotype file exactly as it was *after* messiness was added but *before* realistic population genetics/causal signal was embedded (i.e. still pure noise, but already re-keyed to the phenotype clone_ids and messified). | Lets you compare "messy but random" vs "messy and realistic" genotype data, and recover this intermediate state without re-running the whole generation chain. |
| `data/processed/phenotype_clean.csv`, `data/processed/genotype_clean.csv` | ETL output: normalized `clone_id`, coerced numeric dtypes, duplicates resolved. Fully regenerated by `src/etl_pipeline.py` - never hand-edited. | The strict, model-ready schema `src/capstone_pipeline.py` requires (unique non-null `clone_id`, fully numeric marker columns, numeric targets). |
| `data/processed/etl_report.json` | Machine-readable summary of what the ETL stage changed (rows dropped/merged/coerced), one block per input file. | Quick way to confirm the ETL step is actually doing meaningful cleanup, and to spot a data-quality regression at a glance. |

## `src/` - scripts

### Active pipeline (run in this order when regenerating everything)

| Script | Purpose | Run when |
|---|---|---|
| `generate_messy_raw_data.py` | **(1)** Adds 15 synthetic breeding clones to the commercial phenotype table and injects realistic messiness into both raw CSVs (casing/whitespace noise, alternate missing tokens, percent-as-string values, duplicate rows, malformed genotype calls). Overwrites `data/raw/raw_phenotypes.csv` and `data/raw/raw_genotypes.csv` in place (backs nothing up itself - see `pre_etl_backup/`). | Only if you want to reseed the raw data from the pristine originals in `pre_etl_backup/`. Already run once; do not re-run against the current (already-messy) files or it will double-apply the synthetic clones and messiness. |
| `etl_pipeline.py` | **(2)** Extract-transform-load: reads the raw CSVs (recognizing the messy missing-value tokens), normalizes `clone_id` (trim/collapse whitespace/uppercase), coerces mixed-type and percent-as-string values to numeric, resolves exact and `clone_id`-keyed duplicate rows, and writes `data/processed/phenotype_clean.csv` + `genotype_clean.csv` + `etl_report.json`. | Every time the raw data changes. Run it after step 1 so the genotype simulation has current phenotype values, and again after step 3 to clean the regenerated genotype matrix. |
| `scoring_pipeline.py` | Calculates the five phenotype indices: DRI, EQI, AYI, Market Index, and Parental Value. It uses cleaned source traits and transparent fallback rules for missing values. | Automatically called by `etl_pipeline.py`; computed indices are kept out of `data/raw/` and added to `data/processed/`. |
| `simulate_realistic_genotypes.py` | **(3)** Replaces the genotype matrix with a population-genetics-style simulation: per-marker minor allele frequencies (U-shaped Beta distribution), 6 simulated breeding-program subpopulations with Fst-like drift, linkage-disequilibrium blocks (correlated neighboring markers), and - critically - 25 "causal" markers per modeled trait whose allele frequency is nudged by each *labeled* clone's real phenotype before a stochastic genotype draw. Re-applies the same messiness as step 1 afterward. Backs up the previous genotype file to `pre_realism_backup/` before overwriting. | Whenever you want to change how much genetic signal is embedded (see constants at the top of the file: `CAUSAL_MARKERS_PER_TARGET`, `EFFECT_STRENGTH_RANGE`, `CONDITIONED_TARGETS`) or after re-running step 1 and step 2. |
| `capstone_pipeline.py` | **(4)** The modeling stage. Uses Elastic Net (`ElasticNetCV` + `SelectKBest`) as the primary model and compares it with Ridge (`RidgeCV`) and a compact MLP baseline for the five computed prediction targets `market_index`, `dri`, `eqi`, `parental_value`, and `ayi`. These names are identical in the scored phenotype CSV, simulation, notebooks, and model outputs; Parental Value uses only `parental_value`. Measured yield remains available for scoring but is not a genomic prediction target. Models train on labeled clones only and predict every target for **every** genotyped clone. The comparison uses identical data/CV folds across model families. Writes only `ranked_clones_multi_index.csv` and `model_comparison_all_targets.csv`; the combined ranking includes each target's predicted, observed, and out-of-fold predicted values. | Every time you want fresh predictions/rankings, e.g. after the ETL output changes, or when comparing model families. Runnable directly as a CLI or imported from the notebook or smoke tests. |
| `run_smoke_tests.py` | Lightweight, dependency-free (no pytest) smoke and sanity tests for every script above. Exercises the pure generator functions on small in-memory fixtures (never touches the real `data/raw/` files, to avoid re-mutating already-generated data), actually runs the ETL and modeling stages end to end (safe: both are idempotent and write to temp directories during the test), and import-checks the legacy scripts below without executing their file-mutating `main()`. | Before trusting any pipeline change, and after editing any script in `src/`. Run with `python src\run_smoke_tests.py`; exits non-zero if anything fails. |

## Regenerating phenotype records without changing `raw_phenotypes.csv`

`data/raw/raw_phenotypes.csv` is a **curated** file (non-synthetic rows copied by hand from
`pre_realism_backup/raw_phenotypes_old.csv`, plus literature-generated rows). The three phenotype
generation scripts must never be allowed to leave their output in it. `fill_raw_phenotype_gaps.py`
and `finalize_raw_phenotypes.py` edit `raw_phenotypes.csv` **in place**, so follow these steps:

1. **Back up** the curated file: copy `data/raw/raw_phenotypes.csv` to
   `data/raw/raw_phenotypes_curated_backup.csv`.
2. Run, from the `sweetpotato_capstone_notebook` folder and in this order:
   `python -m src.generate_literature_phenotypes` (writes `raw_phenotypes2.csv`), then copy
   `raw_phenotypes2.csv` over `raw_phenotypes.csv`, then `python -m src.fill_raw_phenotype_gaps`,
   then `python -m src.finalize_raw_phenotypes`.
3. **Move the result aside:** rename the now-overwritten `raw_phenotypes.csv` to
   `data/raw/raw_phenotypes_generated.csv`.
4. **Restore the curated file:** rename `raw_phenotypes_curated_backup.csv` back to `raw_phenotypes.csv`.
5. **Manually copy and paste** only the records you want from `raw_phenotypes_generated.csv` into
   `raw_phenotypes.csv` (same 21 columns, same column order). Do not replace the file.
6. Run `python -m src.etl_pipeline` and check that `data/processed/etl_report.json` still reports
   the expected clone counts.

## Why the genotype data was rebuilt (realism + non-negative R2)

The first version of the genotype matrix was pure independent random 0/1/2 noise per
marker per clone - no minor-allele-frequency structure, no linkage disequilibrium, no
population structure, and (most importantly) **no relationship to any phenotype at all**.
Cross-validated R2 was reliably negative, which was the mathematically correct outcome:
there was nothing for a model to learn.

`simulate_realistic_genotypes.py` fixes this two ways at once:

1. **Realism**: markers get a realistic minor-allele-frequency spectrum, clones are grouped
   into simulated breeding-program subpopulations with their own allele-frequency drift, and
   markers within a linkage-disequilibrium block frequently share genotype calls - much closer
   to how real SNP data looks than independent uniform noise.
2. **Genuine, non-leaky signal**: for each modeled trait, a subset of "causal" markers has its
   allele frequency nudged by each *labeled* clone's standardized phenotype before a stochastic
   `Binomial(2, p)` draw. This embeds a moderate, discretized, noisy dose-response relationship
   - exactly the kind of signal real genomic prediction relies on - without directly leaking the
   target value into a feature. Unlabeled breeding candidates get no such nudge, since their
   phenotype is exactly what the model is supposed to predict.

Historical result: the pure-noise genotype matrix produced cross-validated R2 values
roughly from -0.4 to -0.8, which was expected because the markers contained no phenotype
signal. With the current realistic genotype matrix, the Ridge, Elastic Net, and compact
MLP models can be compared across the four computed prediction targets. For `market_index`,
the comparison is
ElasticNet 0.709, MLP 0.729, and Ridge 0.699.

## `notebooks/`

| Notebook | Purpose |
|---|---|
| `01_genomic_prediction_analysis.ipynb` | The active, reproducible analysis record. Runs the ETL stage, calls `run_capstone_analysis`, displays cross-validated metrics for all four targets, compares model families, charts top predicted *unphenotyped* candidates (change `RANKING_TARGET` to view another target), and shows an observed-vs-cross-validated-predicted diagnostic scatter for labeled clones. |
## `outputs/`

The analysis writes only these two consolidated CSV files:

- `ranked_clones_multi_index.csv` - one row per genotyped clone, containing the predicted,
  observed, and out-of-fold predicted values for each modeled target, plus the training-clone
  indicator. Sort by any `predicted_<target>` column to obtain that target's ranking.
- `model_comparison_all_targets.csv` - model-family cross-validation results for all four
  targets, including CV metrics, sample counts, marker counts, and overfitting-risk indicators.
