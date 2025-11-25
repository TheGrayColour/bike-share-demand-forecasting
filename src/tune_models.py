"""tune_models.py

Randomized hyperparameter search for RandomForest / GradientBoosting models
using TimeSeriesSplit. Saves CV results, best params, and test metrics.

Usage:
    python src/tune_models.py \
        --input data/processed_hour.csv \
        --model rf \
        --n-iter 30 \
        --output-dir results \
        --model-dir models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, period in [("hour", 24), ("dayofweek", 7), ("month", 12), ("dayofyear", 366)]:
        if col in df.columns:
            radians = 2 * np.pi * df[col] / period
            df[f"{col}_sin"] = np.sin(radians)
            df[f"{col}_cos"] = np.cos(radians)
    return df


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def chronological_split(df: pd.DataFrame, target: str, test_size: float):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    feature_cols = [c for c in df.columns if c not in {target, "timestamp"}]
    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df[target],
        test_df[target],
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def param_distributions(model: str):
    if model == "rf":
        return {
            "model__n_estimators": [200, 300, 400, 600, 800],
            "model__max_depth": [10, 15, 20, 30, None],
            "model__max_features": ["sqrt", 0.3, 0.5, 0.7, None],
            "model__min_samples_leaf": [1, 2, 4, 6, 8],
        }
    return {
        "model__n_estimators": [300, 500, 700, 900],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__max_depth": [2, 3, 4, 5],
        "model__subsample": [0.6, 0.8, 1.0],
    }


def build_estimator(model: str, random_state: int):
    if model == "rf":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1)
    return GradientBoostingRegressor(random_state=random_state)


def make_serializable(cv_results: dict) -> dict:
    serializable = {}
    for key, value in cv_results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable


DEFAULT_RANDOM_SEARCH_BUDGET = {"rf": 160, "gbm": 120}


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_cyclical_features(df)

    X_train, X_test, y_train, y_test = chronological_split(df, args.target, args.test_size)
    preprocessor = build_preprocessor(X_train)
    estimator = build_estimator(args.model, args.random_state)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

    tscv = TimeSeriesSplit(n_splits=args.cv_splits)
    n_iter = args.n_iter or DEFAULT_RANDOM_SEARCH_BUDGET[args.model]
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions(args.model),
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        random_state=args.random_state,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_pipeline = search.best_estimator_
    preds = best_pipeline.predict(X_test)
    metrics = regression_metrics(y_test, preds)

    prefix = f"{args.model}"
    cv_results_path = output_dir / f"{prefix}_cv_results.json"
    with cv_results_path.open("w") as f:
        json.dump(make_serializable(search.cv_results_), f, indent=2)

    best_params_path = output_dir / f"{prefix}_best_params.json"
    with best_params_path.open("w") as f:
        json.dump(search.best_params_, f, indent=2)

    metrics_path = output_dir / f"{prefix}_tuned_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    model_path = model_dir / f"{prefix}_tuned_pipeline.joblib"
    joblib.dump(best_pipeline, model_path, compress=3)

    print(f"Best params saved to {best_params_path}")
    print(f"CV results saved to {cv_results_path}")
    print(f"Tuned metrics: {metrics}")
    print(f"Saved tuned model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for bike-share models.")
    parser.add_argument("--input", required=True, help="Processed CSV.")
    parser.add_argument("--output-dir", default="results", help="Directory for metrics and results.")
    parser.add_argument("--model-dir", default="models", help="Directory for serialized pipelines.")
    parser.add_argument("--model", choices=["rf", "gbm"], default="rf")
    parser.add_argument("--target", default="cnt")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="RandomizedSearchCV budget. Defaults to 160 for RF, 120 for GBM.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)

