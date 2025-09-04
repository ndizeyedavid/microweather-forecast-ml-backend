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


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
	if len(req.samples) == 0:
		raise HTTPException(status_code=400, detail="No samples provided")

	# Convert list of samples to DataFrame expected by the pipeline
	data = [s.dict() for s in req.samples]
	df = pd.DataFrame(data)

	# Predict rainfall
	model = get_model()
	try:
		preds = model.predict(df)
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

	# Build enriched response with original fields plus predicted rainfall
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
