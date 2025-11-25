"""rolling_eval.py

Perform rolling-origin evaluation (sliding window) for RandomForest/GBM models.

Usage:
    python src/rolling_eval.py \
        --input data/processed_hour.csv \
        --model rf \
        --train-window 7000 \
        --test-window 168 \
        --step 168 \
        --output results/rolling_eval_rf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
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


def build_model(model: str, random_state: int, overrides: dict | None = None):
    overrides = overrides or {}
    if model == "rf":
        params = {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": random_state,
        }
        params.update(overrides)
        params["random_state"] = random_state
        params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**params)
    params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "random_state": random_state,
    }
    params.update(overrides)
    params["random_state"] = random_state
    return GradientBoostingRegressor(**params)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def load_param_overrides(path: str | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    overrides = {}
    for key, value in data.items():
        if key.startswith("model__"):
            overrides[key.split("model__")[1]] = value
    return overrides


def rolling_windows(df: pd.DataFrame, train_window: int, test_window: int, step: int, strategy: str):
    total = len(df)
    start = train_window
    while start + test_window <= total:
        if strategy == "sliding":
            train_slice = df.iloc[start - train_window : start]
        else:
            train_slice = df.iloc[:start]
        test_slice = df.iloc[start : start + test_window]
        yield train_slice, test_slice
        start += step


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_cyclical_features(df)

    feature_cols = [c for c in df.columns if c not in {"timestamp", args.target}]

    results = []
    preds_records = []
    param_overrides = load_param_overrides(args.param_file)
    estimator_template = build_model(args.model, args.random_state, overrides=param_overrides)

    for idx, (train_slice, test_slice) in enumerate(
        rolling_windows(df, args.train_window, args.test_window, args.step, args.strategy), start=1
    ):
        X_train = train_slice[feature_cols]
        y_train = train_slice[args.target]
        X_test = test_slice[feature_cols]
        y_test = test_slice[args.target]

        preprocessor = build_preprocessor(X_train)
        estimator = estimator_template.__class__(**estimator_template.get_params())
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        metrics = regression_metrics(y_test, preds)
        metrics.update(
            {
                "window": idx,
                "train_start": train_slice["timestamp"].iloc[0].isoformat(),
                "train_end": train_slice["timestamp"].iloc[-1].isoformat(),
                "test_start": test_slice["timestamp"].iloc[0].isoformat(),
                "test_end": test_slice["timestamp"].iloc[-1].isoformat(),
            }
        )
        results.append(metrics)
        preds_records.extend(
            {
                "timestamp": test_slice["timestamp"].iloc[i],
                "y_true": float(y_true),
                "y_pred": float(pred),
                "window": idx,
            }
            for i, (y_true, pred) in enumerate(zip(y_test.values, preds))
        )

    summary = pd.DataFrame(results)
    aggregated = {
        "model": args.model,
        "train_window": args.train_window,
        "test_window": args.test_window,
        "step": args.step,
        "windows_evaluated": len(results),
        "rmse_mean": float(summary["rmse"].mean()),
        "rmse_std": float(summary["rmse"].std()),
        "mae_mean": float(summary["mae"].mean()),
        "mape_mean": float(summary["mape"].mean()),
    }

    output_path = Path(args.output or f"results/rolling_eval_{args.model}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"summary": aggregated, "per_window": results}, f, indent=2)

    preds_path = output_path.with_name(output_path.stem + "_predictions.csv")
    pd.DataFrame(preds_records).to_csv(preds_path, index=False)

    print(f"Saved rolling evaluation summary to {output_path}")
    print(f"Saved rolling predictions to {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling-origin evaluation for bike-share models.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="Optional explicit path for JSON summary.")
    parser.add_argument("--model", choices=["rf", "gbm"], default="rf")
    parser.add_argument("--target", default="cnt")
    parser.add_argument("--train-window", type=int, default=7000)
    parser.add_argument("--test-window", type=int, default=168)
    parser.add_argument("--step", type=int, default=168)
    parser.add_argument(
        "--strategy",
        choices=["sliding", "expanding"],
        default="sliding",
        help="Sliding keeps a fixed-length training window; expanding grows the history.",
    )
    parser.add_argument(
        "--param-file",
        help="Optional JSON of best_params_ (from tuning) to override estimator hyperparameters.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)

