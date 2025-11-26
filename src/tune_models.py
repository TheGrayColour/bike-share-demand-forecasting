"""tune_models.py

Nested cross-validation hyperparameter tuning for RandomForest / GradientBoosting models.
Uses outer CV loop for robust evaluation and inner CV loop for hyperparameter search.

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


def nested_cv_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    random_state: int,
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_iter: int = 120,
) -> tuple[dict, list[dict], dict]:
    """
    Perform nested cross-validation for hyperparameter tuning.
    
    Returns:
        - best_params: Most frequently selected hyperparameters across outer folds
        - outer_fold_results: List of results for each outer fold
        - nested_cv_metrics: Average metrics across outer folds
    """
    outer_cv = TimeSeriesSplit(n_splits=outer_splits)
    inner_cv = TimeSeriesSplit(n_splits=inner_splits)
    
    outer_fold_results = []
    all_best_params = []
    
    print(f"Starting nested CV: {outer_splits} outer folds, {inner_splits} inner folds")
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X), 1):
        print(f"\n--- Outer Fold {fold_idx}/{outer_splits} ---")
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Build preprocessor on training fold only
        preprocessor = build_preprocessor(X_train_fold)
        estimator = build_estimator(model, random_state)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        
        # Inner CV: hyperparameter search
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions(model),
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
            return_train_score=True,
        )
        search.fit(X_train_fold, y_train_fold)
        
        # Evaluate best model on outer validation set
        best_pipeline = search.best_estimator_
        val_preds = best_pipeline.predict(X_val_fold)
        val_metrics = regression_metrics(y_val_fold, val_preds)
        
        fold_result = {
            "fold": fold_idx,
            "best_params": search.best_params_,
            "best_cv_score": float(-search.best_score_),
            "val_metrics": val_metrics,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
        }
        outer_fold_results.append(fold_result)
        all_best_params.append(search.best_params_)
        
        print(f"  Best CV RMSE: {fold_result['best_cv_score']:.2f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.2f}")
        print(f"  Best params: {search.best_params_}")
    
    # Find most common best params (simple voting)
    param_counts = {}
    for params in all_best_params:
        params_key = tuple(sorted(params.items()))
        param_counts[params_key] = param_counts.get(params_key, 0) + 1
    
    most_common_params = max(param_counts.items(), key=lambda x: x[1])[0]
    best_params = dict(most_common_params)
    
    # Average metrics across outer folds
    nested_cv_metrics = {
        "rmse": float(np.mean([r["val_metrics"]["rmse"] for r in outer_fold_results])),
        "mae": float(np.mean([r["val_metrics"]["mae"] for r in outer_fold_results])),
        "mape": float(np.mean([r["val_metrics"]["mape"] for r in outer_fold_results])),
        "r2": float(np.mean([r["val_metrics"]["r2"] for r in outer_fold_results])),
        "rmse_std": float(np.std([r["val_metrics"]["rmse"] for r in outer_fold_results])),
        "mae_std": float(np.std([r["val_metrics"]["mae"] for r in outer_fold_results])),
    }
    
    return best_params, outer_fold_results, nested_cv_metrics


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_cyclical_features(df)

    # Split into train/test (test set is held out completely)
    X_train, X_test, y_train, y_test = chronological_split(df, args.target, args.test_size)
    
    # Perform nested CV on training set
    n_iter = args.n_iter or DEFAULT_RANDOM_SEARCH_BUDGET[args.model]
    best_params, outer_fold_results, nested_cv_metrics = nested_cv_tuning(
        X=X_train,
        y=y_train,
        model=args.model,
        random_state=args.random_state,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_iter=n_iter,
    )
    
    # Train final model on full training set with best params
    print(f"\n--- Training final model on full training set ---")
    preprocessor = build_preprocessor(X_train)
    estimator = build_estimator(args.model, args.random_state)
    
    # Apply best params to estimator
    if args.model == "rf":
        final_estimator = RandomForestRegressor(
            random_state=args.random_state,
            n_jobs=-1,
            **{k.replace("model__", ""): v for k, v in best_params.items()},
        )
    else:  # gbm
        final_estimator = GradientBoostingRegressor(
            random_state=args.random_state,
            **{k.replace("model__", ""): v for k, v in best_params.items()},
        )
    
    final_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", final_estimator)])
    final_pipeline.fit(X_train, y_train)
    
    # Evaluate on held-out test set
    test_preds = final_pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, test_preds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.model}"
    
    # Save nested CV results
    nested_cv_path = output_dir / f"{prefix}_nested_cv_results.json"
    with nested_cv_path.open("w") as f:
        json.dump({
            "outer_fold_results": outer_fold_results,
            "nested_cv_metrics": nested_cv_metrics,
            "best_params": best_params,
        }, f, indent=2)
    
    best_params_path = output_dir / f"{prefix}_best_params.json"
    with best_params_path.open("w") as f:
        json.dump(best_params, f, indent=2)

    # Save test metrics
    metrics_path = output_dir / f"{prefix}_tuned_metrics.json"
    with metrics_path.open("w") as f:
        json.dump({
            "test_metrics": test_metrics,
            "nested_cv_metrics": nested_cv_metrics,
        }, f, indent=2)

    model_path = model_dir / f"{prefix}_tuned_pipeline.joblib"
    joblib.dump(final_pipeline, model_path, compress=3)

    print(f"\n=== Results ===")
    print(f"Nested CV RMSE: {nested_cv_metrics['rmse']:.2f} ± {nested_cv_metrics['rmse_std']:.2f}")
    print(f"Test RMSE: {test_metrics['rmse']:.2f}")
    print(f"Best params: {best_params}")
    print(f"\nSaved nested CV results to {nested_cv_path}")
    print(f"Saved best params to {best_params_path}")
    print(f"Saved test metrics to {metrics_path}")
    print(f"Saved final model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nested cross-validation hyperparameter tuning for bike-share models."
    )
    parser.add_argument("--input", required=True, help="Processed CSV.")
    parser.add_argument("--output-dir", default="results", help="Directory for metrics and results.")
    parser.add_argument("--model-dir", default="models", help="Directory for serialized pipelines.")
    parser.add_argument("--model", choices=["rf", "gbm"], default="rf")
    parser.add_argument("--target", default="cnt")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out as final test set.")
    parser.add_argument(
        "--outer-splits",
        type=int,
        default=5,
        help="Number of outer CV folds for robust evaluation.",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=5,
        help="Number of inner CV folds for hyperparameter search.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="RandomizedSearchCV budget per inner fold. Defaults to 160 for RF, 120 for GBM.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)

