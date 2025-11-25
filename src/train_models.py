"""train_models.py

Train scikit-learn pipelines (feature engineering + estimator) for bike-share demand.

Features:
- ColumnTransformer with numeric scaling + categorical one-hot encoding
- Optional cyclical encodings for hour/day/month
- Chronological train/test split (no leakage)
- Supports RandomForest ("rf") and GradientBoosting ("gbm")
- Saves metrics, prediction plots, and serialized pipelines

Usage:
    python src/train_models.py \
        --input data/processed_hour.csv \
        --output_dir results \
        --model rf \
        --test-size 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for hour, dayofweek, month if present."""
    out = df.copy()
    cycles = [
        ("hour", 24),
        ("dayofweek", 7),
        ("month", 12),
        ("dayofyear", 366),
    ]
    for col, period in cycles:
        if col in out.columns:
            radians = 2 * np.pi * out[col] / period
            out[f"{col}_sin"] = np.sin(radians)
            out[f"{col}_cos"] = np.cos(radians)
    return out


def chronological_split(
    df: pd.DataFrame, target: str, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    feature_cols = [c for c in df.columns if c not in {target, "timestamp"}]
    X_train = train_df[feature_cols].reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_train = train_df[target].reset_index(drop=True)
    y_test = test_df[target].reset_index(drop=True)
    timestamps = test_df["timestamp"].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, timestamps


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols: List[str] = feature_df.select_dtypes(include=[np.number, "float64", "float32", "int64"]).columns.tolist()
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return preprocessor


def build_estimator(model_name: str, args: argparse.Namespace):
    if model_name == "rf":
        estimator = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=args.random_state,
        )
    elif model_name == "gbm":
        estimator = GradientBoostingRegressor(
            n_estimators=args.gbm_n_estimators,
            learning_rate=args.gbm_learning_rate,
            max_depth=args.gbm_max_depth,
            subsample=args.gbm_subsample,
            random_state=args.random_state,
        )
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Use 'rf' or 'gbm'.")

    return estimator


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape), "r2": float(r2)}


def plot_predictions(timestamps: pd.Series, y_true: pd.Series, y_pred: np.ndarray, title: str, output: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, y_true, label="Actual", linewidth=1)
    plt.plot(timestamps, y_pred, label="Predicted", linewidth=1)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Hourly count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def train_pipeline(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_cyclical_features(df)

    X_train, X_test, y_train, y_test, test_timestamps = chronological_split(df, target=args.target, test_size=args.test_size)

    preprocessor = build_preprocessor(X_train)
    estimator = build_estimator(args.model, args)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
    start = perf_counter()
    pipeline.fit(X_train, y_train)
    duration = perf_counter() - start

    y_pred = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{args.model}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    model_path = model_dir / f"{args.model}_pipeline.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    plot_path = output_dir / f"{args.model}_pred_vs_actual.png"
    plot_predictions(
        timestamps=test_timestamps,
        y_true=y_test,
        y_pred=y_pred,
        title=f"{args.model.upper()} predictions vs actual",
        output=plot_path,
    )

    if args.save_predictions:
        preds_df = pd.DataFrame(
            {
                "timestamp": test_timestamps,
                "y_true": y_test,
                "y_pred": y_pred,
            }
        )
        preds_path = output_dir / f"{args.model}_test_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"Saved predictions to {preds_path}")

    print(f"{args.model.upper()} metrics: {metrics}")
    print(f"Training duration: {duration:.2f}s")
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved prediction plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scikit-learn models for bike-share forecasting.")
    parser.add_argument("--input", required=True, help="Processed CSV path (output of prepare.py).")
    parser.add_argument("--output_dir", default="results", help="Directory to store metrics/plots.")
    parser.add_argument("--model_dir", default="models", help="Directory to store serialized pipelines.")
    parser.add_argument("--model", choices=["rf", "gbm"], default="rf", help="Estimator to train.")
    parser.add_argument("--target", default="cnt", help="Target column name.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for chronological test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for estimators.")
    parser.add_argument("--save-predictions", action="store_true", help="Persist test predictions for ensemble use.")
    # RF hyperparameters
    parser.add_argument("--rf-n-estimators", type=int, default=300, help="RandomForest n_estimators.")
    parser.add_argument("--rf-max-depth", type=int, default=20, help="RandomForest max_depth (None for unlimited).")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2, help="RandomForest min_samples_leaf.")
    # GBM hyperparameters
    parser.add_argument("--gbm-n-estimators", type=int, default=600, help="GradientBoosting n_estimators.")
    parser.add_argument("--gbm-learning-rate", type=float, default=0.05, help="GradientBoosting learning rate.")
    parser.add_argument("--gbm-max-depth", type=int, default=4, help="GradientBoosting max_depth for base estimators.")
    parser.add_argument("--gbm-subsample", type=float, default=0.9, help="GradientBoosting subsample fraction.")
    return parser.parse_args()


if __name__ == "__main__":
    train_pipeline(parse_args())

