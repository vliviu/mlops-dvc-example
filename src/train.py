#!/usr/bin/env python3
"""
Robust train.py for mlops-dvc-example

Usage (called by dvc stage):
  python src/train.py <train_csv> <test_csv> <model_out> <metrics_out>

This script:
 - reads params.yaml to get training params and label name (default "quality")
 - validates that the label exists (or tries sensible fallbacks)
 - trains a RandomForestRegressor and writes model + metrics JSON
 - logs params/metrics/model to MLflow
"""
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


def find_label_column(df: pd.DataFrame, preferred_label: str):
    # Exact match
    if preferred_label in df.columns:
        return preferred_label
    # common alternatives
    for alt in ["label", "target", "y", "Quality", "quality_score"]:
        if alt in df.columns:
            return alt
    # try last column if it's numeric
    if len(df.columns) > 0:
        last_col = df.columns[-1]
        if pd.api.types.is_numeric_dtype(df[last_col]):
            return last_col
    return None


def prepare_xy(df: pd.DataFrame, label_col: str):
    df = df.copy()
    # try to coerce object columns to numeric where possible
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            # replace only if coercion yields at least one non-NaN (helps categorical columns)
            if coerced.notna().any():
                df[c] = coerced
    y = df[label_col].astype(float)
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])
    return X, y


def main(train_csv, test_csv, model_out, metrics_out):
    # load params
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise RuntimeError("params.yaml not found in repository root")
    params = yaml.safe_load(params_path.read_text()) or {}
    train_params = params.get("train", {})
    preferred_label = params.get("label", "quality")

    # read data
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # debug: print columns if label missing
    if preferred_label not in train.columns:
        print(f"[WARN] Preferred label '{preferred_label}' not found in train columns.")
        print("Train columns:", list(train.columns))
    if preferred_label not in test.columns:
        print(f"[WARN] Preferred label '{preferred_label}' not found in test columns.")
        print("Test columns:", list(test.columns))

    label_col = find_label_column(train, preferred_label)
    if label_col is None:
        raise RuntimeError(
            "Could not find a suitable label column in train CSV. "
            "Columns found: {}".format(list(train.columns))
        )
    # ensure test has label; try to find equivalent
    if label_col not in test.columns:
        label_col_test = find_label_column(test, preferred_label)
        if label_col_test is None:
            raise RuntimeError(
                f"Label column '{label_col}' found in train but not in test. "
                f"Train cols: {list(train.columns)}; Test cols: {list(test.columns)}"
            )
        label_col = label_col_test

    print(f"[INFO] Using label column: '{label_col}'")

    X_train, y_train = prepare_xy(train, label_col)
    X_test, y_test = prepare_xy(test, label_col)

    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        raise RuntimeError(f"After selecting numeric features, X_train is empty: shape={X_train.shape}")

    # model params with defaults
    n_estimators = int(train_params.get("n_estimators", 10))
    max_depth = train_params.get("max_depth", None)
    max_depth = None if max_depth is None else int(max_depth)
    random_state = int(train_params.get("random_state", 42))

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # MLflow: start run and log params/metrics/model
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("label_column", label_col)
        mlflow.log_param("X_train_shape", f"{X_train.shape}")
        mlflow.log_param("X_test_shape", f"{X_test.shape}")

        # train
        model.fit(X_train, y_train)

        # predict + metrics
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        # log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # mlflow: log the sklearn model artifact (this stores it in MLflow artifacts)
        mlflow.sklearn.log_model(model, artifact_path="model")

    # ensure local output folders exist and write model + metrics for DVC
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)

    # write model for DVC tracking (joblib)
    joblib.dump(model, model_out)

    # write metrics JSON for DVC tracking/consumption by downstream stages
    with open(metrics_out, "w") as f:
        json.dump({"rmse": rmse, "r2": r2, "label_column": label_col}, f)

    print(f"[DONE] Model saved to {model_out}. Metrics written to {metrics_out}. RMSE={rmse:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python src/train.py train.csv test.csv model_out metrics_out")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
