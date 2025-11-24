"""prepare.py

End-to-end data preparation for the bike-share demand project.

Features:
* Flexible timestamp parsing (either `timestamp` column or UCI hour format: dteday + hr)
* Optional hourly resampling to fill gaps / enforce cadence
* Time-based features (hour/day/month/weekend/year)
* Configurable target lags
* Optional weather merge (expects timestamp column)

Usage:
    python src/prepare.py \
        --input /mnt/data/hour.csv \
        --output data/processed_hour.csv \
        --lags 1 3 6 24 \
        --resample \
        --weather data/weather.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV and ensure a datetime `timestamp` column exists."""
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif {"dteday", "hr"}.issubset(df.columns):
        df["dteday"] = pd.to_datetime(df["dteday"])
        df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    else:
        raise ValueError("Input CSV must contain either 'timestamp' or both 'dteday' and 'hr'.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to hourly cadence using forward-fill for missing targets/features."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = (
        df.set_index("timestamp")
        .resample("H")
        .agg({col: "mean" for col in numeric_cols})
        .ffill()
        .reset_index()
    )
    return df


def merge_weather(df: pd.DataFrame, weather_path: str | Path) -> pd.DataFrame:
    """Merge hourly weather CSV on timestamp if provided."""
    weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
    weather = weather.sort_values("timestamp")
    suffix_cols = [c for c in weather.columns if c != "timestamp"]
    weather = weather.rename(columns={col: f"weather_{col}" for col in suffix_cols})
    merged = df.merge(weather, on="timestamp", how="left")
    return merged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived features."""
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    return df


def create_lag_features(df: pd.DataFrame, target: str, lags: Iterable[int]) -> pd.DataFrame:
    """Create lagged versions of the target column."""
    df = df.sort_values("timestamp").copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(args: argparse.Namespace) -> None:
    df = load_csv(args.input)

    if args.resample:
        df = resample_hourly(df)

    if args.weather:
        df = merge_weather(df, args.weather)

    df = add_time_features(df)

    if args.lags:
        df = create_lag_features(df, target=args.target, lags=args.lags)

    df = df.dropna().reset_index(drop=True)
    save_csv(df, args.output)
    print(f"Saved processed data to {args.output}. Rows: {len(df):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare bike-share demand dataset.")
    parser.add_argument("--input", required=True, help="Input CSV path (raw hour data)")
    parser.add_argument("--output", required=True, help="Output processed CSV path")
    parser.add_argument(
        "--lags",
        type=int,
        nargs="*",
        default=[1, 3, 6, 24],
        help="Lag steps (hours) for target column.",
    )
    parser.add_argument(
        "--target",
        default="cnt",
        help="Target column name (default: 'cnt' for UCI dataset).",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample to hourly cadence before feature engineering.",
    )
    parser.add_argument(
        "--weather",
        default=None,
        help="Optional weather CSV with 'timestamp' column to merge.",
    )
    args = parser.parse_args()
    main(args)
