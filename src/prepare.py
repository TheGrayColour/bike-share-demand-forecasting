
"""prepare.py
Simple data preparation utilities for bike-share forecasting.
Usage examples:
  python src/prepare.py --input data/sample_bikeshare.csv --output data/processed_sample.csv --create-lags 1 3 6 24
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def load_csv(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

def resample_hourly(df):
    df = df.set_index('timestamp').resample('H').mean().ffill().reset_index()
    return df

def add_time_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    return df

def create_lag_features(df, col='count', lags=[1,3,6,24]):
    df = df.sort_values('timestamp')
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
        df = create_lag_features(df, lags=args.lags)
    # drop rows with NA created by lags
    df = df.dropna().reset_index(drop=True)
    save_csv(df, args.output)
    print(f"Saved processed data to {args.output}. Rows: {len(df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input CSV path')
    parser.add_argument('--output', required=True, help='output processed CSV path')
    parser.add_argument('--lags', type=int, nargs='*', default=[1,3,6,24], help='lags to create')
    args = parser.parse_args()
    main(args)
