import os
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

FORECAST_STEPS = 7
DEFAULT_INTERVAL = timedelta(minutes=180)
FEATURE_COLUMNS = [
	"rainfall_lag1",
	"pressure_lag1",
	"temperature_lag1",
	"humidity_lag1",
	"hour_sin",
	"hour_cos",
	"dayofyear_sin",
	"dayofyear_cos",
]

TARGET_COLUMNS = ["rainfall", "pressure", "temperature", "humidity"]

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sensor_forecast")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "measurements")
MAX_RECORDS = int(os.getenv("MONGO_MAX_RECORDS", "100"))


_mongo_client: Optional[MongoClient] = None


app = FastAPI(title="Sensor Forecast API", version="4.0.0")

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Add your frontend URL
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class Sample(BaseModel):
	Timestamp: str = Field(..., description="Observation timestamp (ISO or common datetime format)")
	previous_rainfall: float = Field(..., description="Most recent rainfall measurement prior to the prediction timestamp")
	previous_pressure: float = Field(..., description="Most recent pressure measurement prior to the prediction timestamp")
	previous_temperature: float = Field(..., description="Most recent temperature measurement prior to the prediction timestamp")
	previous_humidity: float = Field(..., description="Most recent humidity measurement prior to the prediction timestamp")


class PredictionRequest(BaseModel):
	samples: List[Sample]


class PredictedValues(BaseModel):
	rainfall: float
	pressure: float
	temperature: float
	humidity: float


class PredictionItem(BaseModel):
	Timestamp: str
	predicted: PredictedValues


class PredictionResponse(BaseModel):
	items: List[PredictionItem]


_model = None



def _get_collection():
	global _mongo_client
	if _mongo_client is None:
		_mongo_client = MongoClient(MONGO_URI)
	return _mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]


def _serialize_document(doc: dict) -> dict:
	serialized = dict(doc)
	serialized["_id"] = str(serialized.get("_id"))
	timestamp = serialized.get("timestamp")
	if isinstance(timestamp, datetime):
		serialized["timestamp"] = timestamp.isoformat()
	created_at = serialized.get("created_at")
	if isinstance(created_at, datetime):
		serialized["created_at"] = created_at.isoformat()
	return serialized


def _persist_records(records: List[dict]) -> None:
	if not records:
		return
	collection = _get_collection()
	collection.insert_many(records)


def _enforce_collection_limit(limit: int = MAX_RECORDS) -> None:
	if limit <= 0:
		return
	collection = _get_collection()
	total = collection.count_documents({})
	if total <= limit:
		return
	excess = total - limit
	cursor = collection.find({}, {"_id": 1}).sort("created_at", 1).limit(excess)
	ids_to_delete = [doc["_id"] for doc in cursor]
	if ids_to_delete:
		collection.delete_many({"_id": {"$in": ids_to_delete}})


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


def _resolve_interval(timestamps: pd.Series) -> timedelta:
	if len(timestamps) >= 2:
		diffs = timestamps.sort_values().diff().dropna()
		diffs = diffs[diffs > pd.Timedelta(0)]
		if not diffs.empty:
			return diffs.iloc[-1].to_pytimedelta()
	return DEFAULT_INTERVAL


def _build_feature_row(timestamp: pd.Timestamp, prev_values: dict) -> pd.DataFrame:
	if not isinstance(timestamp, pd.Timestamp):
		timestamp = pd.Timestamp(timestamp)

	hours = timestamp.hour + timestamp.minute / 60.0
	day_of_year = timestamp.dayofyear

	feature = {
		"rainfall_lag1": prev_values["rainfall"],
		"pressure_lag1": prev_values["pressure"],
		"temperature_lag1": prev_values["temperature"],
		"humidity_lag1": prev_values["humidity"],
		"hour_sin": np.sin(2 * np.pi * hours / 24),
		"hour_cos": np.cos(2 * np.pi * hours / 24),
		"dayofyear_sin": np.sin(2 * np.pi * day_of_year / 365.25),
		"dayofyear_cos": np.cos(2 * np.pi * day_of_year / 365.25),
	}

	return pd.DataFrame([feature], columns=FEATURE_COLUMNS)


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
	if len(req.samples) == 0:
		raise HTTPException(status_code=400, detail="No samples provided")

	# Convert list of samples to DataFrame
	data = [s.dict() for s in req.samples]
	df = pd.DataFrame(data)

	required_cols = [
		"Timestamp",
		"previous_rainfall",
		"previous_pressure",
		"previous_temperature",
		"previous_humidity",
	]
	missing_cols = [c for c in required_cols if c not in df.columns]
	if missing_cols:
		raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

	# Parse timestamps and numerical inputs
	df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
	if df["Timestamp"].isna().any():
		raise HTTPException(status_code=400, detail="Some timestamp values could not be parsed. Ensure they use a valid datetime format.")

	for col in ["previous_rainfall", "previous_pressure", "previous_temperature", "previous_humidity"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	if df[["previous_rainfall", "previous_pressure", "previous_temperature", "previous_humidity"]].isna().any().any():
		raise HTTPException(status_code=400, detail="Previous sensor readings must be numeric and non-null.")

	# Sort samples chronologically to simulate progressive forecasting
	df = df.sort_values("Timestamp").reset_index(drop=True)

	interval = _resolve_interval(df["Timestamp"])

	model = get_model()
	request_id = str(uuid4())
	created_at = datetime.utcnow()

	last_pred = None
	last_time = None

	# Walk through provided samples to warm-start the state
	for _, row in df.iterrows():
		prev_values = {
			"rainfall": float(row["previous_rainfall"]),
			"pressure": float(row["previous_pressure"]),
			"temperature": float(row["previous_temperature"]),
			"humidity": float(row["previous_humidity"]),
		}
		features = _build_feature_row(row["Timestamp"], prev_values)
		try:
			pred = model.predict(features)[0]
		except Exception as exc:
			raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
		last_pred = {name: float(value) for name, value in zip(TARGET_COLUMNS, pred)}
		last_time = row["Timestamp"]

	if last_pred is None or last_time is None:
		raise HTTPException(status_code=400, detail="Unable to generate forecasts from provided samples.")

	# Generate progressive forecasts
	items: List[PredictionItem] = []
	current_state = last_pred
	current_time = last_time

	for _ in range(FORECAST_STEPS):
		next_time = current_time + interval
		features = _build_feature_row(next_time, current_state)
		try:
			pred = model.predict(features)[0]
		except Exception as exc:
			raise HTTPException(status_code=400, detail=f"Prediction failed during forecast rollout: {exc}")
		current_state = {name: float(value) for name, value in zip(TARGET_COLUMNS, pred)}
		items.append(
			PredictionItem(
				Timestamp=pd.Timestamp(next_time).isoformat(),
				predicted=PredictedValues(**current_state),
			)
		)
		current_time = next_time

	records: List[dict] = []
	for idx, row in df.iterrows():
		records.append(
			{
				"request_id": request_id,
				"label": "real",
				"sequence": idx,
				"timestamp": pd.Timestamp(row["Timestamp"]).to_pydatetime(),
				"values": {
					"rainfall": float(row["previous_rainfall"]),
					"pressure": float(row["previous_pressure"]),
					"temperature": float(row["previous_temperature"]),
					"humidity": float(row["previous_humidity"]),
				},
				"source": "input",
				"created_at": created_at,
			},
		)

	for step_index, item in enumerate(items, start=1):
		records.append(
			{
				"request_id": request_id,
				"label": "predicted",
				"sequence": step_index,
				"timestamp": pd.Timestamp(item.Timestamp).to_pydatetime(),
				"values": item.predicted.dict(),
				"source": "forecast",
				"created_at": created_at,
			},
		)

	try:
		_persist_records(records)
		_enforce_collection_limit()
	except PyMongoError as exc:
		raise HTTPException(status_code=500, detail=f"Database persistence failed: {exc}")

	return PredictionResponse(items=items)


@app.get("/predictions")
async def list_predictions():
	try:
		docs = list(_get_collection().find().sort("timestamp", 1))
	except PyMongoError as exc:
		raise HTTPException(status_code=500, detail=f"Database error: {exc}")
	return {"items": [_serialize_document(doc) for doc in docs]}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
