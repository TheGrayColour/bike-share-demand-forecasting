
"""baseline.py

Compute the persistence baseline (next hour demand = previous hour) and save metrics.

Usage:
    python src/baseline.py \
        --input data/processed_hour.csv \
        --output results/baseline_persistence_metrics.json \
        --target cnt
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
    df = df.dropna().reset_index(drop=True)
    return df


def evaluate(y_true, y_pred) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def main(args: argparse.Namespace) -> None:
    df = load_data(args.input)
    df_pred = persistence_baseline(df, target_col=args.target)
    metrics = evaluate(df_pred[args.target], df_pred["prediction"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Persistence baseline metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate persistence baseline (hourly).")
    parser.add_argument("--input", required=True, help="Processed CSV with timestamp + target.")
    parser.add_argument("--output", required=True, help="Output JSON path for metrics.")
    parser.add_argument("--target", default="cnt", help="Target column (default: cnt).")
    args = parser.parse_args()
    main(args)
