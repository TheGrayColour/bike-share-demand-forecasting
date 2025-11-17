"""prepare.py
Simple data preparation utilities for bike-share forecasting.
Usage:
  python src/prepare.py --input data/hour.csv --output data/processed_hour.csv --lags 1 3 6 24
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def load_csv(path):
    """Load CSV and ensure a datetime 'timestamp' column exists.

    Supports:
    - files that already contain a 'timestamp' column
    - UCI hour.csv format with 'dteday' and 'hr' columns
    """
    df = pd.read_csv(path)
    # if timestamp present, parse it; else try to build from dteday + hr
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'dteday' in df.columns and 'hr' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])
        # 'hr' is hour of day (0-23); create timestamp as date + hour
        df['timestamp'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')
    else:
        raise ValueError("Input CSV must contain either 'timestamp' or both 'dteday' and 'hr' columns.")
    return df

def resample_hourly(df):
    # Ensure index is timestamp and aggregate numeric columns with hourly mean
    df = df.set_index('timestamp').resample('H').mean().ffill().reset_index()
    return df

def add_time_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    # also add month/year if available
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    return df

def create_lag_features(df, col='cnt', lags=[1,3,6,24]):
    df = df.sort_values('timestamp').copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[col].shift(lag)
    return df

def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main(args):
    df = load_csv(args.input)
    df = resample_hourly(df)
    df = add_time_features(df)
    if args.lags:
        df = create_lag_features(df, col=args.target, lags=args.lags)
    # drop rows with NA created by lags
    df = df.dropna().reset_index(drop=True)
    save_csv(df, args.output)
    print(f"Saved processed data to {args.output}. Rows: {len(df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input CSV path')
    parser.add_argument('--output', required=True, help='output processed CSV path')
    parser.add_argument('--lags', type=int, nargs='*', default=[1,3,6,24], help='lags to create')
    parser.add_argument('--target', default='cnt', help="target column name (default: 'cnt' for UCI dataset)")
    args = parser.parse_args()
    main(args)
