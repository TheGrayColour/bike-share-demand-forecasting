"""run_ablation.py

Run feature ablation experiments for scikit-learn models (RF/GBM).

Each configuration toggles subsets of features (lags, time/cyclical, weather) or
reads a custom definition from a JSON/YAML file. Metrics are saved to
`results/feature_ablation.json` plus an RMSE plot.

Usage:
    python src/run_ablation.py \
        --input data/processed_hour.csv \
        --output results/feature_ablation.json \
        --model rf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
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


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def chronological_split(df: pd.DataFrame, target: str, test_size: float):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    X_train = train_df.drop(columns=[target, "timestamp"], errors="ignore")
    X_test = test_df.drop(columns=[target, "timestamp"], errors="ignore")
    y_train = train_df[target]
    y_test = test_df[target]
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def build_model(model_name: str):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
    raise ValueError(f"Unsupported model '{model_name}'. Only 'rf' is implemented for ablation.")


def feature_configurations(feature_cols: List[str]) -> Dict[str, List[str]]:
    lag_cols = [c for c in feature_cols if c.startswith("lag_")]
    weather_cols = [c for c in feature_cols if c.startswith("weather_")]

    time_candidates = [
        "hour",
        "dayofweek",
        "is_weekend",
        "month",
        "year",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
    ]
    time_cols = [c for c in time_candidates if c in feature_cols]

    base_set = set(feature_cols)
    configs = {
        "full": sorted(base_set),
        "no_weather": sorted(base_set - set(weather_cols)),
        "no_lags": sorted(base_set - set(lag_cols)),
        "lag_only": sorted(set(lag_cols)),
        "lag_time": sorted(set(lag_cols + time_cols)),
    }
    # ensure no empty feature configs
    for name, cols in configs.items():
        if not cols:
            raise ValueError(f"Configuration '{name}' ended up with zero features. Check feature engineering pipeline.")
    return configs


def load_custom_configs(path: Path, available_cols: List[str]) -> Dict[str, List[str]]:
    """Load configs from JSON/YAML file. Supports entries with `columns`, or `include`/`exclude`."""
    try:
        import yaml  # type: ignore
        loader = yaml.safe_load
    except Exception:  # pragma: no cover - yaml optional
        loader = None

    text = path.read_text()
    if path.suffix.lower() in {".yml", ".yaml"} and loader:
        data = loader(text)
    else:
        data = json.loads(text)

    configs = {}
    for entry in data:
        name = entry.get("name")
        if not name:
            raise ValueError("Each config entry must include a 'name'.")
        if "columns" in entry:
            cols = entry["columns"]
        else:
            include = entry.get("include")
            exclude = set(entry.get("exclude", []))
            if include:
                cols = include
            else:
                cols = [c for c in available_cols if c not in exclude]
        missing = set(cols) - set(available_cols)
        if missing:
            raise ValueError(f"Config '{name}' references unknown columns: {missing}")
        configs[name] = cols
    return configs


def build_model(model_name: str, args: argparse.Namespace):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=args.random_state,
        )
    if model_name == "gbm":
        return GradientBoostingRegressor(
            n_estimators=args.gbm_n_estimators,
            learning_rate=args.gbm_learning_rate,
            max_depth=args.gbm_max_depth,
            subsample=args.gbm_subsample,
            random_state=args.random_state,
        )
    raise ValueError(f"Unsupported model '{model_name}'.")


def run_ablation(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_cyclical_features(df)

    X_all, X_test_full, y_all, y_test_full = chronological_split(df, args.target, args.test_size)

    available_cols = X_all.columns.tolist()
    if args.config_file:
        configs = load_custom_configs(Path(args.config_file), available_cols)
    else:
        configs = feature_configurations(available_cols)

    results = []
    for model_name in args.models:
        print(f"=== Model: {model_name.upper()} ===")
        for name, cols in configs.items():
            print(f"Running config: {name} ({len(cols)} features)")
            X_train_cfg = X_all[cols]
            X_test_cfg = X_test_full[cols]

            preprocessor = build_preprocessor(X_train_cfg)
            estimator = build_model(model_name, args)
            pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
            pipeline.fit(X_train_cfg, y_all)
            preds = pipeline.predict(X_test_cfg)
            metrics = regression_metrics(y_test_full, preds)
            metrics.update({"config": name, "n_features": len(cols), "model": model_name})
            results.append(metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablation metrics to {output_path}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("rmse")
    pivot = df_results.pivot(index="config", columns="model", values="rmse")
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=True)
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("RMSE")
    ax.set_title("Feature ablation")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plot_path = output_path.parent / "feature_ablation_rmse.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved ablation plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature ablation experiments.")
    parser.add_argument("--input", required=True, help="Processed CSV (output of prepare.py)")
    parser.add_argument("--output", default="results/feature_ablation.json", help="Metrics JSON output path")
    parser.add_argument("--target", default="cnt", help="Target column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--models", nargs="+", default=["rf"], choices=["rf", "gbm"], help="Models to evaluate.")
    parser.add_argument("--config-file", help="Optional JSON/YAML file describing feature configs.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--rf-n-estimators", type=int, default=300, help="RandomForest n_estimators.")
    parser.add_argument("--rf-max-depth", type=int, default=20, help="RandomForest max_depth.")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2, help="RandomForest min_samples_leaf.")
    parser.add_argument("--gbm-n-estimators", type=int, default=600, help="GradientBoosting n_estimators.")
    parser.add_argument("--gbm-learning-rate", type=float, default=0.05, help="GradientBoosting learning rate.")
    parser.add_argument("--gbm-max-depth", type=int, default=4, help="GradientBoosting max_depth.")
    parser.add_argument("--gbm-subsample", type=float, default=0.9, help="GradientBoosting subsample.")
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation(parse_args())

