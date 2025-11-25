"""prepare.py

End-to-end data preparation for the bike-share demand project.

Outputs include:
- `timestamp` (datetime, sorted)
- Calendar features: `hour`, `dayofweek`, `is_weekend`, `month`, `year`, `dayofyear`
- Optional cyclical encodings: `<col>_sin`, `<col>_cos`
- Rolling aggregates of the target (defaults: 3h and 24h)
- Lagged target columns (`lag_{k}`)
- Weather columns prefixed with `weather_` when merged plus interaction terms
- Holiday indicators (`is_holiday`, `is_holiday_eve`, `is_holiday_followup`)

Usage:
    python src/prepare.py \
        --input /mnt/data/hour.csv \
        --output data/processed_hour.csv \
        --lags 1 3 6 24 \
        --resample-frequency H \
        --add-cyclical \
        --rolling-windows 3 24 \
        --weather data/weather.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


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


def resample_hourly(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Resample to a regular cadence using forward-fill for missing targets/features."""
    freq = frequency.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    agg_map = {col: "mean" for col in numeric_cols}
    resampled = (
        df.set_index("timestamp")
        .resample(freq)
        .agg(agg_map)
        .ffill()
        .reset_index()
    )
    return resampled


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
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    return df


def add_cyclical_features(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Add sine/cosine representations for periodic columns."""
    df = df.copy()
    periods = {"hour": 24, "dayofweek": 7, "month": 12, "dayofyear": 366}
    for col in columns:
        if col in df.columns and col in periods:
            radians = 2 * np.pi * df[col] / periods[col]
            df[f"{col}_sin"] = np.sin(radians)
            df[f"{col}_cos"] = np.cos(radians)
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate rows that fall on/around US federal holidays."""
    df = df.copy()
    calendar = USFederalHolidayCalendar()
    start = df["timestamp"].min().normalize()
    end = df["timestamp"].max().normalize()
    holidays = calendar.holidays(start=start, end=end)
    normalized = df["timestamp"].dt.normalize()
    df["is_holiday"] = normalized.isin(holidays).astype(int)
    df["is_holiday_eve"] = normalized.isin(holidays - pd.Timedelta(days=1)).astype(int)
    df["is_holiday_followup"] = normalized.isin(holidays + pd.Timedelta(days=1)).astype(int)
    return df


def create_lag_features(df: pd.DataFrame, target: str, lags: Iterable[int]) -> pd.DataFrame:
    """Create lagged versions of the target column."""
    df = df.sort_values("timestamp").copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str, windows: Iterable[int]) -> pd.DataFrame:
    """Add rolling mean features for the target column."""
    df = df.sort_values("timestamp").copy()
    for window in windows:
        df[f"rolling_{window}"] = df[target].rolling(window=window, min_periods=1).mean()
    return df


def add_weather_interactions(df: pd.DataFrame, base_cols: Sequence[str]) -> pd.DataFrame:
    """Create simple interaction terms between weather columns and key temporal flags."""
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    available_bases = [c for c in base_cols if c in df.columns]
    if not weather_cols or not available_bases:
        return df

    df = df.copy()
    for wcol in weather_cols:
        for bcol in available_bases:
            df[f"{wcol}_x_{bcol}"] = df[wcol] * df[bcol]
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(args: argparse.Namespace) -> None:
    df = load_csv(args.input)
    if args.resample_frequency and not args.skip_resample:
        df = resample_hourly(df, args.resample_frequency)

    if args.weather:
        df = merge_weather(df, args.weather)

    df = add_time_features(df)
    if not args.skip_holidays:
        df = add_holiday_features(df)

    if args.add_cyclical:
        df = add_cyclical_features(df, columns=["hour", "dayofweek", "month", "dayofyear"])

    if args.lags:
        df = create_lag_features(df, target=args.target, lags=args.lags)
    if args.rolling_windows:
        df = add_rolling_features(df, target=args.target, windows=args.rolling_windows)

    if args.weather and not args.disable_weather_interactions:
        df = add_weather_interactions(df, base_cols=["hour", "is_weekend", "is_holiday"])

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
        "--resample-frequency",
        default="H",
        help="Pandas offset alias for resampling cadence (e.g., 'H', '30T'). Leave empty to skip.",
    )
    parser.add_argument(
        "--skip-resample",
        action="store_true",
        help="Disable resampling even if --resample-frequency is set.",
    )
    parser.add_argument(
        "--weather",
        default=None,
        help="Optional weather CSV with 'timestamp' column to merge.",
    )
    parser.add_argument(
        "--add-cyclical",
        dest="add_cyclical",
        action="store_true",
        help="Add sine/cosine encodings for hour/dayofweek/month/dayofyear (default).",
    )
    parser.add_argument(
        "--skip-cyclical",
        dest="add_cyclical",
        action="store_false",
        help="Disable cyclical encodings if desired.",
    )
    parser.set_defaults(add_cyclical=True)
    parser.add_argument(
        "--skip-holidays",
        action="store_true",
        help="Disable holiday indicator engineering.",
    )
    parser.add_argument(
        "--disable-weather-interactions",
        action="store_true",
        help="Skip weather interaction terms even if weather data provided.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="*",
        default=[3, 24],
        help="Rolling window sizes (in hours) for target moving averages.",
    )
    args = parser.parse_args()
    main(args)
