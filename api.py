import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

app = FastAPI(title="Rainfall Prediction API", version="1.1.0")

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Add your frontend URL
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class Sample(BaseModel):
	Station_Name: str = Field(..., description="Station name")
	Lat: float
	Lon: float
	Elev: float
	Year: int
	Month: int
	Day: int
	TMPMAX: Optional[float] = None
	TMPMIN: Optional[float] = None
	Rainfall_lag1: Optional[float] = None  # Previous day's rainfall


class PredictionRequest(BaseModel):
	samples: List[Sample]


class PredictionItem(BaseModel):
	Station_Name: str
	Lat: float
	Lon: float
	Elev: float
	Year: int
	Month: int
	Day: int
	TMPMAX: Optional[float]
	TMPMIN: Optional[float]
	Predicted_Rainfall_mm: float


class PredictionResponse(BaseModel):
	items: List[PredictionItem]


_model = None


def get_model():
	global _model
	if _model is None:
		if not os.path.exists(MODEL_PATH):
			raise RuntimeError(
				f"Model file not found at {MODEL_PATH}. Train a model first (run train_model.py)."
			)
		_model = joblib.load(MODEL_PATH)
	return _model


@app.get("/health")
async def health():
	return {"status": "ok"}


def create_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
	"""Create the same features as in training script"""
	df = df.copy()
	
	# Create rolling averages for temperature (if we have enough data)
	if len(df) >= 3:
		df['TMPMAX_avg3'] = df['TMPMAX'].rolling(window=3, min_periods=1).mean()
		df['TMPMIN_avg3'] = df['TMPMIN'].rolling(window=3, min_periods=1).mean()
	else:
		df['TMPMAX_avg3'] = df['TMPMAX']
		df['TMPMIN_avg3'] = df['TMPMIN']
	
	# Create temperature difference
	df['TMP_diff'] = df['TMPMAX'] - df['TMPMIN']
	
	# Create seasonal features
	df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
	df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
	
	# Create day of year
	df['DayOfYear'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.dayofyear
	df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
	df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
	
	return df


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
	if len(req.samples) == 0:
		raise HTTPException(status_code=400, detail="No samples provided")

	# Convert list of samples to DataFrame
	data = [s.dict() for s in req.samples]
	df = pd.DataFrame(data)

	# Create features for prediction (same as training)
	df = create_features_for_prediction(df)
	
	# Ensure we have the Rainfall_lag1 column (previous day's rainfall)
	if 'Rainfall_lag1' not in df.columns or df['Rainfall_lag1'].isna().all():
		# If no lagged rainfall provided, set to 0 (no previous rainfall)
		df['Rainfall_lag1'] = 0.0

	# Predict rainfall
	model = get_model()
	try:
		preds = model.predict(df)
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

	# Build enriched response
	items: List[PredictionItem] = []
	for sample_dict, pred in zip(data, preds):
		item = PredictionItem(
			Predicted_Rainfall_mm=float(pred),
			**sample_dict,
		)
		items.append(item)

	return PredictionResponse(items=items)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
