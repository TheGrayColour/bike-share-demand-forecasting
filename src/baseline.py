
"""baseline.py

Compute the persistence baseline (next hour demand = previous hour) on a chronological split and save metrics.

Usage:
    python src/baseline.py \
        --input data/processed_hour.csv \
        --output results/baseline_persistence_metrics.json \
        --target cnt \
        --test-size 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def persistence_baseline(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df["prediction"] = df[target_col].shift(1)
    return df


def evaluate(y_true, y_pred) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def main(args: argparse.Namespace) -> None:
    df = load_data(args.input)
    df_pred = persistence_baseline(df, target_col=args.target)
    split_idx = int(len(df_pred) * (1 - args.test_size))
    test_block = df_pred.iloc[split_idx:].reset_index(drop=True)
    test_block = test_block.dropna(subset=["prediction"])
    metrics = evaluate(test_block[args.target], test_block["prediction"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Persistence baseline metrics:", metrics)
    if args.save_predictions:
        preds = test_block[["timestamp", args.target, "prediction"]].rename(
            columns={args.target: "y_true", "prediction": "y_pred"}
        )
        preds.to_csv(Path(args.output).with_name("baseline_persistence_predictions.csv"), index=False)
        print("Saved baseline predictions alongside metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate persistence baseline (hourly).")
    parser.add_argument("--input", required=True, help="Processed CSV with timestamp + target.")
    parser.add_argument("--output", required=True, help="Output JSON path for metrics.")
    parser.add_argument("--target", default="cnt", help="Target column (default: cnt).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for evaluation (chronological).")
    parser.add_argument("--save-predictions", action="store_true", help="Store CSV of persistence predictions for test period.")
    args = parser.parse_args()
    main(args)
