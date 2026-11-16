# DTProfiler

A machine learning pipeline for inferring SQL data types (VARCHAR, NUMBER, BOOLEAN, TIME) of CSV columns from structural statistics and FastText column-name embeddings.

Paper: *DTProfiler: SQL Data Type Inference For Automated Data Pipelines Using Machine Learning* [paper in preparation]

## Quick Look: Why Regex Typing Fails

[`examples/toy.csv`](examples/toy.csv) is a tiny table with identifier-like columns. The regex labeler used for initial labeling (`utils/regex_data_type_detector.py`) assigns `NUMBER` to `customer_id`, `zip_code`, and `phone`, although casting them to a numeric type would drop leading zeros (e.g. `0000141`, `07030`). These are the cases that the column-name embeddings and manual label review are meant to correct.

```bash
python -c "import pandas as pd; from utils.regex_data_type_detector import RegexDataTypeDetector as R; df = pd.read_csv('examples/toy.csv', dtype=str); d = R(); [print(c, '->', d.detect_column_type(df[c])) for c in df]"
```

## Setup

Python 3.11. Use a virtual environment: the pinned versions matter (`torch==2.2.2` needs `numpy<2`).

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name dtprofiler   # select this kernel in Jupyter / your IDE
```

## Run Order

Entry points are numbered in the order they should be run. (`utils/` and `model/` hold importable modules and are not run directly.)

| # | File | Purpose |
|---|---|---|
| 01 | [`01_get_data_filepaths.py`](01_get_data_filepaths.py) | Collect CSV paths from a source directory |
| 02 | [`02_run_data_pipeline.py`](02_run_data_pipeline.py) | Extract statistical and embedding features |
| 03 | [`03_data_analysis.ipynb`](03_data_analysis.ipynb) | Exploratory data analysis |
| 04 | [`04_data_preparation.ipynb`](04_data_preparation.ipynb) | Clean features, table-grouped train/val/test split, PCA |
| 05 | [`05_model_pipeline.ipynb`](05_model_pipeline.ipynb) | Train/evaluate models; multi-seed, grouped-split, ablation and PCA analyses |

To reproduce the results you only need `data/` and notebook 05. The split it uses ships in `data/splits/table_seed42/`.

## Results

### Evaluation protocol

A column-level random split is optimistic: on the original split about 96 % of test columns had a sibling column from the same CSV in the training set. All results below therefore use **grouped** splits, where every column of a table (or every table of a data source) lands in exactly one partition:

- **Unseen tables:** table-grouped 70/15/15 splits. No test column shares a CSV with training.
- **Unseen sources:** source-grouped splits. No test column shares a dataset with training; this is the closest setting to profiling a brand-new database.

Each setting uses 5 different splits × 3 model seeds. Hyper-parameters (NN learning rate and batch size, ensemble weight) are chosen on validation data only.

### Headline: mean ± sd over 5 splits × 3 seeds (notebook section 15)

| Model | Unseen tables: Accuracy (%) | Unseen tables: Weighted F1 | Unseen tables: G-mean | Unseen sources: Accuracy (%) | Unseen sources: Weighted F1 | Unseen sources: G-mean |
|---|---|---|---|---|---|---|
| pandas dtype inference | 88.66 ± 2.41 | 0.860 ± 0.031 | 0.000 | 90.33 ± 4.06 | 0.889 ± 0.049 | 0.000 |
| Decision Tree | 92.03 ± 3.08 | 0.920 ± 0.031 | 0.913 ± 0.031 | 86.82 ± 8.14 | 0.867 ± 0.078 | 0.764 ± 0.125 |
| Random Forest | 95.25 ± 1.43 | 0.953 ± 0.014 | 0.944 ± 0.023 | 93.61 ± 3.90 | 0.936 ± 0.039 | 0.872 ± 0.071 |
| Modular NN (z-scored inputs) | 96.46 ± 1.71 | 0.966 ± 0.017 | 0.968 ± 0.025 | 93.96 ± 3.32 | 0.942 ± 0.031 | 0.931 ± 0.064 |
| **RF + NN ensemble** | **97.16 ± 1.18** | **0.972 ± 0.012** | **0.979 ± 0.014** | **95.64 ± 1.83** | **0.957 ± 0.018** | **0.946 ± 0.038** |

pandas never predicts a temporal type (dates stay `object`), so its G-mean is 0 by construction; compare it on accuracy and F1.

How to read the table:

- **Most of the spread comes from the split, not from training.** Changing which tables are held out moves G-mean 3–5× more than changing the model seed (variance decomposition in section 11). A single split can land anywhere in these ranges.
- **The ensemble's main advantage is stability.** Its mean is about 0.01 G-mean above the NN alone, which is within one split-sd, but its spread across splits is roughly half the NN's.
- **Unseen sources are harder.** G-mean drops by about 3 points from the unseen-tables setting to the unseen-sources setting, and the spread doubles.
- **Statistics and name embeddings are complementary** (section 13). On the shipped split, statistics alone give G-mean 0.966 and embeddings alone 0.81; combined they reach 0.986.

### Single split (`data/splits/table_seed42`, 5 model seeds; sections 5–8)

The shipped split has 2,049 / 545 / 594 train/val/test columns from 288 / 61 / 51 tables. On it, the models score:

| Model | Accuracy (%) | Weighted F1 | G-mean |
|---|---|---|---|
| pandas dtype inference | 89.23 | 0.857 | 0.000 |
| Decision Tree | 95.42 ± 1.34 | 0.954 ± 0.014 | 0.956 ± 0.030 |
| Random Forest | 97.04 ± 0.35 | 0.971 ± 0.004 | 0.979 ± 0.003 |
| Modular NN (z-scored inputs) | 98.18 ± 0.14 | 0.982 ± 0.001 | 0.986 ± 0.001 |
| RF + NN ensemble | 98.22 ± 0.19 | 0.982 ± 0.002 | 0.987 ± 0.001 |

These numbers are at the top of the cross-split range. The ± here only reflects training noise (seeds), not which tables happen to be in the test set, so report the headline table above.

### Notes

- **Inputs to the NN are standardized** (scaler fitted on the training split only). `USE_SCALING = False` in section 3.1 reproduces the original unscaled NN, which is much weaker (G-mean 0.885 on the shipped split); section 12 compares both.
- **PCA uses 25 components per token** everywhere (fitted on training rows only, `random_state=42`). Section 14 is a sensitivity check: on one split the RF's validation G-mean ranges from 0.961 to 0.984 across 3–50 components.
- **Column names still repeat across tables** (e.g. `created_at`): 56 % of test columns on the shipped split have a same-named column in training. Part of the remaining performance comes from familiar names.
- **The original column-level random split** from the first version of the paper is kept in `data/X_*.csv` / `data/y_*.csv`. Set `SPLIT_NAME = "paper"` in section 3 of notebook 05 to run on it.

### Runtime and caching

On one laptop CPU:

- **NN hyper-parameter search (section 5.4):** about 10 minutes. All 10 configurations are trained with one seed, then the `TOP_K = 3` best get 4 more seeds (22 runs instead of 50; `TOP_K = None` runs the full grid). Every run is cached in `results/grid_cache/`, so an interrupted search resumes where it stopped.
- **Extended evaluation (sections 10–15):** about 30 minutes. Results are cached as JSON in `results/<split>_lr<lr>_bs<bs>_scaled<bool>/` and loaded by default; set `RUN_EXPERIMENTS = True` in section 9 to recompute them, or `QUICK = True` for a fast smoke test (writes to `results_quick/`).
- Figures are written to `figures/`.

---

## Building the Classifier from Scratch

### Step 1 — Download Datasets

Refer to [`DATA_SOURCES.md`](DATA_SOURCES.md) for the full list of data sources and [`DATA.md`](DATA.md) for download instructions.

---

### Step 2 — Collect CSV File Paths

Use the helper script to collect all CSV file paths from your dataset directory. The resulting file list is consumed by the data pipeline in Step 3.

```bash
python 01_get_data_filepaths.py --source_dir ../path/to/your/data
```

This writes all discovered `.csv` file paths to `data/data_source_files.txt` (one path per line; override with `--output`) for use in the next step.

---

### Step 3 — Run the Data Pipeline

**Prerequisites:** Download `crawl-300d-2M-subword.vec` from [FastText](https://fasttext.cc/docs/en/english-vectors.html) and place it in the `model/` directory.

Run the pipeline to extract features from all collected CSV files:

```bash
python 02_run_data_pipeline.py data/data_source_files.txt
```

> **Note:** The pipeline currently only supports CSV files that use a comma (`,`) as the delimiter.

---

### Step 4 — Review and Correct Data Type Labels

Open the extracted features file and manually review the assigned data types. Update any labels that are semantically incorrect based on the column content. This step ensures label quality before model training.

---

### Step 5 — Exploratory Data Analysis

Open and run [`03_data_analysis.ipynb`](03_data_analysis.ipynb). This end-to-end notebook covers:

1. Load & merge feature data
2. Null value handling
3. Data distributions (labels, characters, lengths, entropy)
4. Correlation matrix
5. PCA on word embeddings
6. Embedding model (FastText OOV handling)
7. Visualizations (neighbors, clustering, score distributions)
8. Data source distribution (VizNet / Kaggle / GitHub / data.gov breakdown)

---

### Step 6 — Prepare Dataset for Training

Open and run [`04_data_preparation.ipynb`](04_data_preparation.ipynb). This notebook covers:

1. Load data
2. Null value handling
3. Drop correlated features
4. Grouped train/val/test split (70/15/15, stratified by label with `StratifiedGroupKFold`)
5. PCA on word embeddings (25 components, fitted on training rows only)
6. Save the split to `data/splits/<SPLIT_MODE>_seed<SPLIT_SEED>/`, including `groups_*.csv` with each row's source table

Split settings in section 4:

| Setting | Values | Meaning |
|---|---|---|
| `SPLIT_MODE` | `"table"` (default), `"source"`, `"random"` | Keep whole tables / whole data sources in one partition, or split columns at random |
| `SPLIT_SEED` | `42` | Seed for the group assignment |
| `VAL_FRAC`, `TEST_FRAC` | `0.15`, `0.15` | Approximate fractions (tables differ in size) |

The notebook asserts that no table appears in two partitions and warns if a class has fewer than 5 rows in val or test.

---

### Step 7 — Train and Evaluate Models

Open and run [`05_model_pipeline.ipynb`](05_model_pipeline.ipynb). Set `SPLIT_NAME` in section 3 to the folder written by notebook 04 (default `table_seed42`). The notebook covers:

1. Load libraries
2. Training pipelines for tree and neural-network models
3. Load data (and 3.1: standardize NN inputs, `USE_SCALING`)
4. Initialize models and class weights
5. Train models: reference DT/RF runs, feature importance, NN hyper-parameter search (5.4), and multi-seed evaluation of DT, RF, NN and the RF+NN ensemble (5.5)
6. Compare results (hyper-parameter heatmaps, pandas baseline, mean ± sd bar charts, confusion matrices summed over seeds)
7. Sub-network contributions (statistics-only vs. name-only NN)
8. Robustness over random seeds
9. Setup for the extended evaluation (`RUN_EXPERIMENTS` / `QUICK` flags)
10. Seed variance and bootstrap confidence intervals on the selected split
11. Leakage diagnostic and evaluation on random, table-grouped and source-grouped splits, with a split-vs-seed variance decomposition
12. Effect of standardizing the NN inputs
13. Feature-group ablation (statistics vs. column-name embeddings, per-class recall)
14. Sensitivity to the number of PCA components
15. **Headline results** on unseen tables and unseen sources

---

## Project Structure

```
├── 01_get_data_filepaths.py     # Step 2: collect CSV paths
├── 02_run_data_pipeline.py      # Step 3: feature extraction
├── 03_data_analysis.ipynb       # Step 5: EDA
├── 04_data_preparation.ipynb    # Step 6: grouped split + PCA
├── 05_model_pipeline.ipynb      # Step 7: training, evaluation, extended analyses
├── data/
│   ├── features_*.csv           # Extracted features (input to notebook 04)
│   ├── splits/table_seed42/     # Table-grouped split used by notebook 05
│   └── X_*.csv, y_*.csv         # Original column-level random split (SPLIT_NAME = "paper")
├── results/
│   ├── grid_cache/              # Cached NN hyper-parameter runs (section 5.4)
│   └── table_seed42_lr0.005_bs1024_scaledTrue/   # Cached results of sections 10-15 (JSON)
├── figures/                     # Figures written by notebook 05
├── examples/                    # Toy CSV illustrating identifier-vs-number ambiguity
├── logs/                        # Pipeline run logs
├── model/                       # Model definitions (+ FastText vectors, not shipped)
│   ├── col_name_neural_network.py
│   ├── modular_neural_network_model.py
│   ├── neural_network_model.py
│   ├── stats_neural_network.py
│   └── tree_model.py
├── utils/                       # Importable pipeline modules
│   ├── backoff_embedding_generator.py
│   ├── data_pipeline.py
│   ├── regex_data_type_detector.py
│   └── statistical_features_extractor.py
├── requirements.txt
├── DATA_SOURCES.md              # Dataset sources and licenses
└── DATA.md                      # Dataset reproduction instructions
```

## Dataset License

[![CC BY 4.0][cc-by-shield]][cc-by]

The derived feature files in the `data/` directory are offered under a
[Creative Commons Attribution 4.0 International License][cc-by], **with one exception**:
rows derived from `data_source_17` (CC BY-SA 4.0; 150 rows) and `data_source_24`
(CC BY-NC 4.0; 37 rows) remain subject to the terms of their upstream licenses
(see [`DATA.md`](DATA.md), *License Notes*). The `Filepath` column identifies the source of each row.
No raw cell values are distributed.

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Citation

```bibtex
@misc{dtprofiler2026,
  title  = {DTProfiler: SQL Data Type Inference For Automated Data Pipelines Using Machine Learning},
  author = {[YOUR FULL NAME]},
  year   = {2026},
  note   = {arXiv: TODO}
}
```
