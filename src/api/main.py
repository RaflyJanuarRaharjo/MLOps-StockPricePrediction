# src/api/main.py
# ============================================================
# FastAPI Backend - AAPL Stock Price Prediction
# Rafly Januar Raharjo - 235150201111011 | MLOps Kelas B
# ============================================================

import os
import glob
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- Konfigurasi ---
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME  = os.getenv("MODEL_NAME", "AAPL-RF-Production")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_7", "MA_14", "MA_30", "RSI_14",
    "MACD", "Signal", "Hist",
    "BB_upper", "BB_lower", "Daily_Return",
    "Close_lag1", "Close_lag2", "Close_lag5", "Vol_MA_7"
]

# --- Setup MLflow ---
mlflow.set_tracking_uri(MLFLOW_URI)

# --- FastAPI App ---
app = FastAPI(
    title       = "AAPL Stock Price Prediction API",
    description = "API prediksi harga penutupan saham AAPL menggunakan Random Forest Regressor",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# --- Load Model ---
model = None

def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        print(f"Model {MODEL_NAME}@{MODEL_ALIAS} berhasil dimuat!")
    except Exception as e:
        print(f"Gagal load model: {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()

# --- Schema ---
class PredictionRequest(BaseModel):
    open:         float
    high:         float
    low:          float
    close:        float
    volume:       float
    ma_7:         float
    ma_14:        float
    ma_30:        float
    rsi_14:       float
    macd:         float
    signal:       float
    hist:         float
    bb_upper:     float
    bb_lower:     float
    daily_return: float
    close_lag1:   float
    close_lag2:   float
    close_lag5:   float
    vol_ma_7:     float

class PredictionResponse(BaseModel):
    prediction:  float
    model_name:  str
    model_alias: str
    timestamp:   str
    message:     str

# --- Endpoints ---
@app.get("/")
def root():
    return {
        "message"    : "AAPL Stock Price Prediction API",
        "version"    : "1.0.0",
        "model"      : f"{MODEL_NAME}@{MODEL_ALIAS}",
        "mlflow_uri" : MLFLOW_URI,
        "status"     : "running"
    }

@app.get("/health")
def health():
    return {
        "status"      : "healthy",
        "model_loaded": model is not None,
        "timestamp"   : datetime.now().isoformat()
    }

@app.get("/model-info")
def model_info():
    return {
        "model_name"  : MODEL_NAME,
        "model_alias" : MODEL_ALIAS,
        "mlflow_uri"  : MLFLOW_URI,
        "features"    : FEATURE_COLS,
        "target"      : "Close T+1 (harga penutupan besok)"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    features = pd.DataFrame([{
        "Open"        : request.open,
        "High"        : request.high,
        "Low"         : request.low,
        "Close"       : request.close,
        "Volume"      : request.volume,
        "MA_7"        : request.ma_7,
        "MA_14"       : request.ma_14,
        "MA_30"       : request.ma_30,
        "RSI_14"      : request.rsi_14,
        "MACD"        : request.macd,
        "Signal"      : request.signal,
        "Hist"        : request.hist,
        "BB_upper"    : request.bb_upper,
        "BB_lower"    : request.bb_lower,
        "Daily_Return": request.daily_return,
        "Close_lag1"  : request.close_lag1,
        "Close_lag2"  : request.close_lag2,
        "Close_lag5"  : request.close_lag5,
        "Vol_MA_7"    : request.vol_ma_7
    }])

    prediction = float(model.predict(features)[0])

    return PredictionResponse(
        prediction  = round(prediction, 2),
        model_name  = MODEL_NAME,
        model_alias = MODEL_ALIAS,
        timestamp   = datetime.now().isoformat(),
        message     = f"Prediksi harga penutupan AAPL besok: ${prediction:.2f}"
    )

@app.get("/predict-latest")
def predict_latest():
    """Prediksi menggunakan data terbaru dari file processed."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    files = sorted(glob.glob("data/processed/aapl_features_*.csv"))
    if not files:
        raise HTTPException(status_code=404, detail="Data processed tidak ditemukan")

    df      = pd.read_csv(files[-1], index_col="Date", parse_dates=True).dropna()
    latest  = df[FEATURE_COLS].tail(1)
    pred    = float(model.predict(latest)[0])
    date    = str(latest.index[0].date())

    return {
        "input_date" : date,
        "prediction" : round(pred, 2),
        "message"    : f"Berdasarkan data {date}, prediksi harga AAPL besok: ${pred:.2f}",
        "model"      : f"{MODEL_NAME}@{MODEL_ALIAS}"
    }
