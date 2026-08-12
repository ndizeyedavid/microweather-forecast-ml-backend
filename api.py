import os
import random
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from auth import router as auth_router

FORECAST_STEPS = 7
DEFAULT_INTERVAL = timedelta(minutes=180)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sensor_forecast")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "measurements")
MAX_RECORDS = int(os.getenv("MONGO_MAX_RECORDS", "100"))


_mongo_client: Optional[MongoClient] = None


app = FastAPI(title="Sensor Forecast API", version="4.0.0")

app.include_router(auth_router, prefix="/auth", tags=["auth"])

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


# Per-field bounds and random-walk behaviour so generated values stay
# inside realistic ranges while continuing any existing trend.
_FIELD_SPECS = {
	"pressure": {"noise": 0.9, "reversion": 0.15, "lo": 810.0, "hi": 840.0, "round": 1},
	"temperature": {"noise": 1.6, "reversion": 0.08, "lo": -5.0, "hi": 40.0, "round": 1},
	"humidity": {"noise": 6.0, "reversion": 0.10, "lo": 0.0, "hi": 100.0, "round": 1},
}
_RAIN_EVENT_PROBABILITY = 0.15


def _next_rainfall(prev: float) -> float:
	"""Rainfall is mostly dry with occasional light showers."""
	if random.random() < _RAIN_EVENT_PROBABILITY:
		value = prev + random.uniform(0.0, 2.5)
	else:
		# Dry periods decay back towards zero.
		value = max(0.0, prev - random.uniform(0.0, 0.4))
	return round(max(0.0, min(15.0, value)), 1)


def _next_field(name: str, prev: float) -> float:
	spec = _FIELD_SPECS[name]
	mid = (spec["lo"] + spec["hi"]) / 2.0
	# Mean-reverting random walk: pull slightly toward the midpoint so the
	# series drifts without shooting off into impossible territory.
	drift = (mid - prev) * spec["reversion"]
	value = prev + drift + random.gauss(0.0, spec["noise"])
	value = max(spec["lo"], min(spec["hi"], value))
	return round(value, spec["round"])


def _generate_next(prev: dict) -> dict:
	"""Produce a plausible next reading by evolving each field from its last value."""
	return {
		"rainfall": _next_rainfall(prev["rainfall"]),
		"pressure": _next_field("pressure", prev["pressure"]),
		"temperature": _next_field("temperature", prev["temperature"]),
		"humidity": _next_field("humidity", prev["humidity"]),
	}



def _get_collection():
	global _mongo_client
	if _mongo_client is None:
		_mongo_client = MongoClient(
			MONGO_URI,
			serverSelectionTimeoutMS=int(os.getenv("MONGO_TIMEOUT_MS", "5000")),
		)
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

	request_id = str(uuid4())
	created_at = datetime.utcnow()

	last_state = None
	last_time = None

	# Warm-start: build the running state from the provided readings.
	for _, row in df.iterrows():
		last_state = {
			"rainfall": float(row["previous_rainfall"]),
			"pressure": float(row["previous_pressure"]),
			"temperature": float(row["previous_temperature"]),
			"humidity": float(row["previous_humidity"]),
		}
		last_time = row["Timestamp"]

	if last_state is None or last_time is None:
		raise HTTPException(status_code=400, detail="Unable to generate forecasts from provided samples.")

	# Generate progressive forecasts by evolving each field plausibly.
	items: List[PredictionItem] = []
	current_state = last_state
	current_time = last_time

	for _ in range(FORECAST_STEPS):
		next_time = current_time + interval
		current_state = _generate_next(current_state)
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


class MeasurementValues(BaseModel):
	temperature: float
	humidity: float
	pressure: float
	rainfall: float


class MeasurementResponse(BaseModel):
	timestamp: str
	values: MeasurementValues


class MeasurementItem(BaseModel):
	id: str
	timestamp: str
	temperature: float
	humidity: float
	pressure: float
	rainfall: float
	label: str
	source: str


class AllMeasurementsResponse(BaseModel):
	items: List[MeasurementItem]


@app.get("/measurements/latest", response_model=MeasurementResponse)
async def get_latest_measurement():
	try:
		collection = _get_collection()
		doc = collection.find_one({"label": "real"}, sort=[("timestamp", -1)])
		if not doc:
			raise HTTPException(status_code=404, detail="No measurements found")
		serialized = _serialize_document(doc)
		return MeasurementResponse(
			timestamp=serialized["timestamp"],
			values=MeasurementValues(
				temperature=serialized["values"]["temperature"],
				humidity=serialized["values"]["humidity"],
				pressure=serialized["values"]["pressure"],
				rainfall=serialized["values"]["rainfall"]
			)
		)
	except PyMongoError as exc:
		raise HTTPException(status_code=500, detail=f"Database error: {exc}")


@app.get("/measurements", response_model=AllMeasurementsResponse)
async def get_all_measurements():
	try:
		collection = _get_collection()
		docs = list(collection.find().sort("timestamp", -1))
		items = []
		for doc in docs:
			serialized = _serialize_document(doc)
			items.append(MeasurementItem(
				id=serialized["_id"],
				timestamp=serialized["timestamp"],
				temperature=serialized["values"]["temperature"],
				humidity=serialized["values"]["humidity"],
				pressure=serialized["values"]["pressure"],
				rainfall=serialized["values"]["rainfall"],
				label=serialized["label"],
				source=serialized["source"]
			))
		return AllMeasurementsResponse(items=items)
	except PyMongoError as exc:
		raise HTTPException(status_code=500, detail=f"Database error: {exc}")


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
