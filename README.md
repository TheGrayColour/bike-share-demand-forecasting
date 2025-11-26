# Bike Share Demand Forecasting

Hourly bike-share demand forecasting project built on the UCI Bike Sharing Dataset. The repository contains preparation scripts, modeling pipelines (Ridge/RF/GBM/LSTM), ablation experiments, notebooks, and reporting assets.

---

## Data Source

- Original dataset: [UCI Bike Sharing](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) (`hour.csv`).
- The raw CSV is **not committed**. Download it locally (or reuse `/mnt/data/hour.csv` in this environment) and pass the path via `RAW_DATA`.
- Generated files (e.g., `data/processed_hour.csv`) are reproducible and ignored by git; use `make prepare` or the `src/prepare.py` script to recreate them.

---

## Environment Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Choose the `Python (bikevenv)` kernel in Jupyter after activating the virtual environment.

---

## Makefile Targets

The Makefile encodes the core workflow. Override variables as needed, e.g. `make prepare RAW_DATA=/path/to/hour.csv`.

| Target | Description |
| --- | --- |
| `make prepare` | Run `src/prepare.py` with hourly resampling, lags (`1/3/6/24`), rolling stats (`3/24h`), holidays, weather interactions, and cyclical encodings to produce `data/processed_hour.csv`. Automatically excludes `registered` and `casual` features to prevent data leakage. |
| `make train` | Train both RandomForest and GradientBoosting pipelines; saves metrics, test predictions, and serialized pipelines. |
| `make tune` | Launch nested cross-validation hyperparameter tuning (outer CV: 5 folds, inner CV: 5 folds with RandomizedSearchCV, RF=160 draws, GBM=120 per inner fold). Persists nested CV results, best parameters, and tuned models under `results/` & `models/`. |
| `make rolling-eval` | Run the sliding-window retrain/eval script for RF+GBM; outputs JSON summaries and per-window predictions. |
| `make ensemble` | Solve for optimal ensemble weights given saved prediction CSVs and write `results/ensemble_metrics.json`. |
| `make ablation` | Execute `src/run_ablation.py` to compare feature subsets; writes `results/feature_ablation.json` + plot. |
| `make notebooks` | Executes the reproducible notebooks (`EDA`, `baseline_eval`, `modeling`) headlessly via `jupyter nbconvert`. |
| `make clean` | Remove generated `.executed.ipynb` files. |

Key variables (override per invocation or set in shell):

```makefile
RAW_DATA ?= /mnt/data/hour.csv
PROCESSED ?= data/processed_hour.csv
RESULTS ?= results
MODELS ?= models
PYTHON ?= py  # Uses Windows Python launcher; change to 'python' on macOS/Linux
```

---

## Manual Commands (if not using Make)

### Prepare Data
```bash
py src/prepare.py \
    --input /mnt/data/hour.csv \
    --output data/processed_hour.csv \
    --resample-frequency H \
    --lags 1 3 6 24 \
    --rolling-windows 3 24
```
*Note: `registered` and `casual` features are automatically excluded to prevent data leakage (they sum to the target `cnt`).*

### Train Models
```bash
py src/train_models.py --input data/processed_hour.csv --output_dir results --model_dir models --model rf
py src/train_models.py --input data/processed_hour.csv --output_dir results --model_dir models --model gbm
```

### Baseline & Ablation
```bash
py src/baseline.py --input data/processed_hour.csv --output results/baseline_persistence_metrics.json
py src/run_ablation.py --input data/processed_hour.csv --output results/feature_ablation.json
```

### Hyperparameter Tuning (Nested CV)
```bash
py src/tune_models.py --input data/processed_hour.csv --model rf --output-dir results --model-dir models
py src/tune_models.py --input data/processed_hour.csv --model gbm --output-dir results --model-dir models
# Nested CV results land in results/rf_nested_cv_results.json (or gbm equivalent)
# Note: Tuned models may overfit to CV folds; untuned models often generalize better
```

### Rolling-Origin Evaluation
```bash
py src/rolling_eval.py \
    --input data/processed_hour.csv \
    --model rf \
    --train-window 7000 \
    --test-window 168 \
    --step 168 \
    --strategy sliding \
    --output results/rolling_eval_rf.json
```

### Ensemble Optimization
```bash
py src/ensemble_opt.py \
    --predictions results/rf_test_predictions.csv \
    --predictions results/gbm_test_predictions.csv \
    --output results/ensemble_metrics.json
```

### LSTM / Ensemble Notebook
- Open `notebooks/lstm_experiment.ipynb`, select the project kernel, and run top-to-bottom to train the sequence model and optional ensemble.

---

## Generated Artifacts

- `results/*.json`: Metrics for persistence, Ridge, RF, GBM, LSTM, ensemble, ablation configs. Nested CV results in `*_nested_cv_results.json`.
- `results/*.png`: Plots for EDA, model comparisons, prediction vs actual, feature importance, ablation bar chart, ensemble vs baseline comparisons.
- `models/*.joblib`: Compressed scikit-learn pipelines (RF/GBM). *Note: Large model files are gitignored; regenerate via `make train`.*
- `models/lstm_best.pt`: Best-performing LSTM checkpoint (from notebook). *Note: Large model files are gitignored.*

---

## Repository Layout

- `src/` – preparation, baseline, training, and ablation scripts.
- `notebooks/` – EDA, baseline evaluation, modeling workflow, LSTM experiment.
- `results/` – metrics/plots generated by scripts/notebooks.
- `models/` – serialized model artifacts.
- `report/` – report/slides (placeholder).
- `Makefile` – reproducible commands for each phase.

---

## Reproducibility Checklist

- `.venv` + `pip install -r requirements.txt` succeeds.
- `make prepare RAW_DATA=/path/to/hour.csv` regenerates processed data.
- `make train` produces `results/rf_metrics.json`, `results/gbm_metrics.json`, and corresponding `.joblib` files.
- `make ablation` updates `results/feature_ablation.json` and `results/feature_ablation_rmse.png`.
- Notebooks execute top-to-bottom using the project kernel (headless via `make notebooks` or interactively in Jupyter).