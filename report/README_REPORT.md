
# Report Outline

This living document tracks the modeling enhancements that should be highlighted in the final report.

## Required Sections
- Title, authors, dataset
- Introduction & motivation
- Data description and preprocessing
- Methods (baseline, sklearn model, optional LSTM)
- Experiments and results (tables and figures)
- Discussion, limitations, future work
- References

## Recent Updates

### Feature Engineering
- Rolling aggregates at 3h and 24h horizons are part of the default preprocessing pipeline.
- US federal holiday flags (`is_holiday`, `is_holiday_eve`, `is_holiday_followup`) capture abrupt demand shifts.
- Weather interactions (e.g., `weather_temp_x_is_weekend`) are generated automatically once weather data is merged.
- Day-of-year cyclical encodings augment previously available hour/day-of-week/month cycles.

### Hyperparameter Search (Nested Cross-Validation)
- `src/tune_models.py` implements **nested cross-validation** for robust hyperparameter tuning:
  - **Outer CV loop** (default 5 folds): Evaluates model performance on held-out validation sets
  - **Inner CV loop** (default 5 folds): Performs RandomizedSearchCV for hyperparameter selection within each outer fold
  - Default search budget: 160 iterations for RF, 120 for GBM per inner fold
- Outputs:
  - `results/{model}_nested_cv_results.json`: Complete nested CV results including per-fold metrics and best params
  - `results/{model}_best_params.json`: Most frequently selected hyperparameters across outer folds
  - `results/{model}_tuned_metrics.json`: Both nested CV metrics (average ± std) and final test set metrics
- This approach prevents overfitting to CV folds and provides more realistic performance estimates.

### Rolling-Origin Evaluation
- Script: `python src/rolling_eval.py --input data/processed_hour.csv --model rf --train-window 7000 --test-window 168 --step 168 --strategy sliding`.
- Supports optional expanding windows and consumption of tuned hyperparameters via `--param-file results/rf_best_params.json`.
- Outputs summary metrics plus per-window predictions under `results/rolling_eval_<model>.json` and `_predictions.csv`.
- Use these artifacts to discuss temporal robustness and variance across windows in the report.

### Ensemble Tuning
- `src/ensemble_opt.py` ingests aligned prediction CSVs (`results/*_test_predictions.csv`) and solves a constrained optimization (non-negative weights summing to 1) that minimizes validation RMSE.
- The script chronologically splits the predictions, learns weights on the validation slice, and reports validation/evaluation metrics alongside the learned weights.
- Cite `results/ensemble_metrics.json` for the tuned ensemble performance rather than static weights like `[0.4, 0.4, 0.2]`.
