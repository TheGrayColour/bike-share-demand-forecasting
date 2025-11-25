"""ensemble_opt.py

Optimize ensemble weights for saved prediction CSVs (e.g., rf/gbm/lstm test predictions).

Each CSV must contain columns: timestamp, y_true, y_pred. The script aligns on timestamp,
solves a non-negative least squares problem (no intercept), projects onto the probability
simplex, and reports metrics.

Usage:
    python src/ensemble_opt.py \
        --predictions results/rf_test_predictions.csv \
        --predictions results/gbm_test_predictions.csv \
        --output results/ensemble_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
def regression_metrics(y_true, y_pred):
    errors = y_pred - y_true
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = np.mean(np.abs(errors) / denom)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def load_predictions(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns={"y_pred": f"pred_{name}"})
    return df[["timestamp", "y_true", f"pred_{name}"]]


def solve_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve for non-negative weights that sum to 1 using clipped least squares."""
    n_models = X.shape[1]
    if n_models == 1:
        return np.array([1.0])

    solution, *_ = np.linalg.lstsq(X, y, rcond=None)
    weights = np.clip(solution, 0, None)
    if weights.sum() == 0:
        weights = np.ones(n_models) / n_models
    return weights / weights.sum()


def main(args: argparse.Namespace) -> None:
    frames = []
    model_names = []
    for entry in args.predictions:
        path = Path(entry)
        name = path.stem.replace("_test_predictions", "")
        frames.append(load_predictions(path, name))
        model_names.append(name)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["timestamp", "y_true"], how="inner")

    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    X = merged[pred_cols].values
    y = merged["y_true"].values

    split_idx = max(1, int(len(X) * args.val_fraction))
    split_idx = min(split_idx, len(X) - 1) if len(X) > 1 else 1
    X_val = X[:split_idx]
    y_val = y[:split_idx]
    X_eval = X[split_idx:] if len(X) > split_idx else X_val
    y_eval = y[split_idx:] if len(y) > split_idx else y_val

    weights = solve_weights(X_val, y_val)

    val_preds = X_val @ weights
    val_metrics = regression_metrics(y_val, val_preds)
    if len(X_eval) == len(X_val) and np.array_equal(X_eval, X_val):
        ensemble_preds = val_preds
        eval_metrics = val_metrics
    else:
        ensemble_preds = X_eval @ weights
        eval_metrics = regression_metrics(y_eval, ensemble_preds)

    metrics = {
        "validation": val_metrics,
        "evaluation": eval_metrics,
        "weights": {model_names[i]: float(weights[i]) for i in range(len(weights))},
        "val_fraction": args.val_fraction,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    preds_full = X @ weights
    preds_df = merged[["timestamp", "y_true"]].copy()
    preds_df["ensemble_pred"] = preds_full
    preds_df.to_csv(output_path.with_name("ensemble_predictions.csv"), index=False)

    print(f"Ensemble metrics saved to {output_path}")
    print(f"Weights: {metrics['weights']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ensemble weights for saved predictions.")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Path to prediction CSV (timestamp,y_true,y_pred). Use multiple times.",
    )
    parser.add_argument("--output", default="results/ensemble_metrics.json")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.6,
        help="Fraction of rows (chronologically) used to fit ensemble weights.",
    )
    args = parser.parse_args()
    main(args)

