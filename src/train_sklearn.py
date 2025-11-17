
"""train_sklearn.py
Train a simple RandomForest on engineered features.
Usage:
  python src/train_sklearn.py --input data/processed_sample.csv --model-path models/rf_model.joblib
"""
import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def prepare_features(df, target='count'):
    # use lag_* and time features if available
    feature_cols = [c for c in df.columns if c.startswith('lag_') or c in ['hour','dayofweek','is_weekend','temperature']]
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y

def train_and_save(X, y, model_path):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

def evaluate(model, X, y):
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    return {'rmse': float(rmse), 'mae': float(mae)}

def main(args):
    df = load_data(args.input)
    X, y = prepare_features(df, target=args.target)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    model = train_and_save(X, y, args.model_path)
    metrics = evaluate(model, X, y)
    print('Train metrics (in-sample):', metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='processed CSV path')
    parser.add_argument('--model-path', default='models/rf_model.joblib', help='output model path')
    parser.add_argument('--target', default='count', help='target column name')
    args = parser.parse_args()
    main(args)
