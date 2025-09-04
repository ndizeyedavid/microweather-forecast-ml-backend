import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "Rainfall"
MODEL_PATH = "model.joblib"


def load_data(csv_path: str) -> pd.DataFrame:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found at: {csv_path}")
	# Keep empty strings as NaN
	df = pd.read_csv(csv_path)
	return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Ensure expected columns exist
    required_cols = [
        "Station_Name", "Lat", "Lon", "Elev", "Year", "Month", "Day",
        "TMPMAX", "TMPMIN", TARGET_COLUMN
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create additional features for better predictions
    df = df.copy()
    
    # Create lagged rainfall feature (previous day's rainfall)
    df['Rainfall_lag1'] = df.groupby('Station_Name')[TARGET_COLUMN].shift(1)
    
    # Create rolling averages for temperature
    df['TMPMAX_avg3'] = df.groupby('Station_Name')['TMPMAX'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['TMPMIN_avg3'] = df.groupby('Station_Name')['TMPMIN'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    # Create temperature difference
    df['TMP_diff'] = df['TMPMAX'] - df['TMPMIN']
    
    # Create seasonal features
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Create day of year
    df['DayOfYear'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.dayofyear
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

    # Create features and target
    feature_cols = [
        "Station_Name", "Lat", "Lon", "Elev", "Year", "Month", "Day",
        "TMPMAX", "TMPMIN", "Rainfall_lag1", "TMPMAX_avg3", "TMPMIN_avg3",
        "TMP_diff", "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"
    ]
    
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
	# Identify column types
	categorical_cols = ["Station_Name"]
	numeric_cols = [
		c for c in X.columns
		if c not in categorical_cols
	]

	categorical_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore")),
	])

	numeric_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("categorical", categorical_transformer, categorical_cols),
			("numeric", numeric_transformer, numeric_cols),
		]
	)

	# Use Ridge regression for better generalization
	from sklearn.linear_model import Ridge
	model = Ridge(alpha=1.0)

	pipeline = Pipeline(steps=[
		("preprocessor", preprocessor),
		("model", model),
	])

	return pipeline


def train_and_save(csv_path: str, model_path: str = MODEL_PATH) -> None:
	df = load_data(csv_path)
	X, y = build_features(df)

	# Some rows may have target missing; drop them for training
	mask = ~pd.isna(y)
	X_train = X.loc[mask]
	y_train = y.loc[mask]

	print(f"Training on {len(X_train)} samples with {len(X_train.columns)} features")
	print(f"Features: {list(X_train.columns)}")

	pipeline = build_pipeline(X_train)
	pipeline.fit(X_train, y_train)

	# Quick fit quality check on training data (since dataset is small)
	pred_train = pipeline.predict(X_train)
	r2 = r2_score(y_train, pred_train)
	mae = mean_absolute_error(y_train, pred_train)
	print(f"Training R2: {r2:.3f}, MAE: {mae:.3f}")
	print(f"Target range: {y_train.min():.2f} - {y_train.max():.2f}")
	print(f"Prediction range: {pred_train.min():.2f} - {pred_train.max():.2f}")

	joblib.dump(pipeline, model_path)
	print(f"Model saved to {model_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train linear regression model for rainfall prediction")
	parser.add_argument("--csv", default="DATA.csv", help="Path to input CSV file")
	parser.add_argument("--out", default=MODEL_PATH, help="Path to save trained model")
	args = parser.parse_args()

	train_and_save(args.csv, args.out)
