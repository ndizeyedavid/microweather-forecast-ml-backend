import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = ["rainfall", "pressure", "temperature", "humidity"]
MODEL_PATH = "model.joblib"


def load_data(csv_path: str) -> pd.DataFrame:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found at: {csv_path}")
	# Keep empty strings as NaN
	df = pd.read_csv(csv_path)
	return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature matrix and labels from the simplified sensor dataset."""
    required_cols = ["Timestamp", "rainfall", "pressure", "temperature", "humidity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        raise ValueError("Some timestamp values could not be parsed. Ensure they follow a valid datetime format.")

    # Sort chronologically to create lag-based features safely
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Time-based cyclical features
    hours = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
    day_of_year = df["Timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Lagged sensors to capture short-term dynamics
    df["rainfall_lag1"] = df["rainfall"].shift(1)
    df["pressure_lag1"] = df["pressure"].shift(1)
    df["temperature_lag1"] = df["temperature"].shift(1)
    df["humidity_lag1"] = df["humidity"].shift(1)

    feature_cols = [
        "rainfall_lag1",
        "pressure_lag1",
        "temperature_lag1",
        "humidity_lag1",
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]

    X = df[feature_cols]
    y = df[TARGET_COLUMNS]
    return X, y


def build_pipeline() -> Pipeline:
	return Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
		("model", Ridge(alpha=1.0)),
	])


def train_and_save(csv_path: str, model_path: str = MODEL_PATH) -> None:
	df = load_data(csv_path)
	X, y = build_features(df)

	# Remove rows where features or targets contain NaNs (e.g., from lag creation)
	feature_mask = ~X.isna().any(axis=1)
	target_mask = ~y.isna().any(axis=1)
	valid_mask = feature_mask & target_mask
	X = X.loc[valid_mask]
	y = y.loc[valid_mask]

	print(f"Training on {len(X)} samples with {len(X.columns)} features")
	print(f"Features: {list(X.columns)}")
	print(f"Targets: {list(y.columns)}")

	pipeline = build_pipeline()
	pipeline.fit(X, y)

	# Quick fit quality check on training data (since dataset is small)
	pred_train = pipeline.predict(X)
	r2 = r2_score(y, pred_train)
	mae = mean_absolute_error(y, pred_train)
	print(f"Training R2 (uniform avg): {r2:.3f}, MAE (uniform avg): {mae:.3f}")
	for idx, column in enumerate(y.columns):
		target_min = y[column].min()
		target_max = y[column].max()
		pred_min = pred_train[:, idx].min()
		pred_max = pred_train[:, idx].max()
		mae_col = mean_absolute_error(y[column], pred_train[:, idx])
		r2_col = r2_score(y[column], pred_train[:, idx])
		print(
			f"  {column}: R2={r2_col:.3f}, MAE={mae_col:.3f},"
			f" target range={target_min:.2f}-{target_max:.2f},"
			f" prediction range={pred_min:.2f}-{pred_max:.2f}"
		)

	joblib.dump(pipeline, model_path)
	print(f"Model saved to {model_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train multi-output regression model for sensor forecasting")
	parser.add_argument("--csv", default="dataset.csv", help="Path to input CSV file")
	parser.add_argument("--out", default=MODEL_PATH, help="Path to save trained model")
	args = parser.parse_args()

	train_and_save(args.csv, args.out)
