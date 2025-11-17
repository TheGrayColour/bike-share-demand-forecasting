
"""baseline.py
Compute persistence baseline (predict next hour = last hour) and report RMSE/MAE.
Usage:
  python src/baseline.py --input data/sample_bikeshare.csv --output results/baseline_metrics.json
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def persistence_baseline(df, target_col='count'):
    df = df.copy()
    df['pred'] = df[target_col].shift(1)  # predict next hour = previous hour
    df = df.dropna().reset_index(drop=True)
    return df

def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'rmse': float(rmse), 'mae': float(mae)}

def main(args):
    df = load_data(args.input)
    df_pred = persistence_baseline(df, target_col=args.target)
    metrics = evaluate(df_pred[args.target], df_pred['pred'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Persistence baseline metrics:", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input CSV path')
    parser.add_argument('--output', required=True, help='output JSON path')
    parser.add_argument('--target', default='count', help='target column name')
    args = parser.parse_args()
    main(args)
