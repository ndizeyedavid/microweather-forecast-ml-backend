RUN: uvicorn api:app --host 0.0.0.0 --port 8000
TRAIN: python train_model.py --csv DATA.csv --out model.joblib
