# src/api/main.py
# ============================================================
# FastAPI Backend - AAPL Stock Price Prediction
# LK-11: Ditambahkan endpoint /metrics untuk Prometheus
# Rafly Januar Raharjo - 235150201111011 | MLOps Kelas B
# ============================================================

import os
import glob
import time
import psutil
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

# --- Konfigurasi ---
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME  = "AAPL-RF-Production"
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

# ============================================================
# PROMETHEUS METRICS (LK-11)
# ============================================================

# Jumlah total request per endpoint dan status
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total jumlah request ke API',
    ['method', 'endpoint', 'status']
)

# Latensi inferensi dalam detik
REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds',
    'Latensi request API dalam detik',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Nilai prediksi harga AAPL (untuk deteksi data drift)
PREDICTION_VALUE = Gauge(
    'aapl_prediction_price_usd',
    'Nilai prediksi harga penutupan AAPL dalam USD'
)

# Total prediksi yang sudah dilakukan
PREDICTION_COUNT = Counter(
    'aapl_predictions_total',
    'Total jumlah prediksi yang dilakukan',
    ['endpoint']
)

# Metrik sistem
CPU_USAGE     = Gauge('system_cpu_usage_percent',    'CPU usage persen')
MEMORY_USAGE  = Gauge('system_memory_usage_bytes',   'Memory usage dalam bytes')
MEMORY_PERCENT = Gauge('system_memory_usage_percent', 'Memory usage persen')

# Status model
MODEL_LOADED = Gauge(
    'model_loaded_status',
    'Status model: 1 = loaded, 0 = not loaded'
)

# ============================================================
# FastAPI App
# ============================================================
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
        MODEL_LOADED.set(1)
        print(f"[INFO] Model {MODEL_NAME}@{MODEL_ALIAS} berhasil dimuat!")
    except Exception as e:
        MODEL_LOADED.set(0)
        print(f"[WARN] Gagal load model: {e}")
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

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    REQUEST_COUNT.labels(method="GET", endpoint="/", status="200").inc()
    return {
        "message"    : "AAPL Stock Price Prediction API",
        "version"    : "1.0.0",
        "model"      : f"{MODEL_NAME}@{MODEL_ALIAS}",
        "mlflow_uri" : MLFLOW_URI,
        "status"     : "running"
    }

@app.get("/health")
def health():
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
    return {
        "status"      : "healthy",
        "model_loaded": model is not None,
        "timestamp"   : datetime.now().isoformat()
    }

@app.get("/model-info")
def model_info():
    REQUEST_COUNT.labels(method="GET", endpoint="/model-info", status="200").inc()
    return {
        "model_name"  : MODEL_NAME,
        "model_alias" : MODEL_ALIAS,
        "mlflow_uri"  : MLFLOW_URI,
        "features"    : FEATURE_COLS,
        "target"      : "Close T+1 (harga penutupan besok)"
    }

@app.get("/metrics")
def metrics():
    """
    Endpoint scraping Prometheus.
    Prometheus membaca metrik dari sini setiap 10-15 detik.
    """
    # Update metrik sistem setiap kali di-scrape
    CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    mem = psutil.virtual_memory()
    MEMORY_USAGE.set(mem.used)
    MEMORY_PERCENT.set(mem.percent)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    start = time.time()
    try:
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

        # Update Prometheus metrics
        PREDICTION_VALUE.set(prediction)
        PREDICTION_COUNT.labels(endpoint="/predict").inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()

        return PredictionResponse(
            prediction  = round(prediction, 2),
            model_name  = MODEL_NAME,
            model_alias = MODEL_ALIAS,
            timestamp   = datetime.now().isoformat(),
            message     = f"Prediksi harga penutupan AAPL besok: ${prediction:.2f}"
        )

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        elapsed = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed)

@app.get("/predict-latest")
def predict_latest():
    """Prediksi menggunakan data terbaru dari file processed."""
    if model is None:
        REQUEST_COUNT.labels(method="GET", endpoint="/predict-latest", status="503").inc()
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    start = time.time()
    try:
        files = sorted(glob.glob("data/processed/aapl_features_*.csv"))
        if not files:
            raise HTTPException(status_code=404, detail="Data processed tidak ditemukan")

        df     = pd.read_csv(files[-1], index_col="Date", parse_dates=True).dropna()
        latest = df[FEATURE_COLS].tail(1)
        pred   = float(model.predict(latest)[0])
        date   = str(latest.index[0].date())

        # Update Prometheus metrics
        PREDICTION_VALUE.set(pred)
        PREDICTION_COUNT.labels(endpoint="/predict-latest").inc()
        REQUEST_COUNT.labels(method="GET", endpoint="/predict-latest", status="200").inc()

        return {
            "input_date" : date,
            "prediction" : round(pred, 2),
            "message"    : f"Berdasarkan data {date}, prediksi harga AAPL besok: ${pred:.2f}",
            "model"      : f"{MODEL_NAME}@{MODEL_ALIAS}"
        }

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method="GET", endpoint="/predict-latest", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        elapsed = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/predict-latest").observe(elapsed)