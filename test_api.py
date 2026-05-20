# test_api.py
# ============================================================
# Pengujian Endpoint API Prediksi AAPL
# Rafly Januar Raharjo - 235150201111011 | MLOps Kelas B
# ============================================================

import requests
import pandas as pd
import glob
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test endpoint health."""
    print("="*55)
    print("  TEST 1: Health Check")
    print("="*55)
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status Code : {r.status_code}")
    print(f"Response    : {json.dumps(r.json(), indent=2)}")
    print()

def test_root():
    """Test endpoint root."""
    print("="*55)
    print("  TEST 2: Root Endpoint")
    print("="*55)
    r = requests.get(f"{BASE_URL}/")
    print(f"Status Code : {r.status_code}")
    print(f"Response    : {json.dumps(r.json(), indent=2)}")
    print()

def test_model_info():
    """Test endpoint model info."""
    print("="*55)
    print("  TEST 3: Model Info")
    print("="*55)
    r = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code : {r.status_code}")
    print(f"Response    : {json.dumps(r.json(), indent=2)}")
    print()

def test_predict_latest():
    """Test prediksi menggunakan data terbaru."""
    print("="*55)
    print("  TEST 4: Predict Latest")
    print("="*55)
    r = requests.get(f"{BASE_URL}/predict-latest")
    print(f"Status Code : {r.status_code}")
    print(f"Response    : {json.dumps(r.json(), indent=2)}")
    print()

def test_predict_manual():
    """Test prediksi dengan input manual."""
    print("="*55)
    print("  TEST 5: Predict Manual Input")
    print("="*55)

    # Ambil data sample dari file processed
    files = sorted(glob.glob("data/processed/aapl_features_*.csv"))
    if not files:
        print("File processed tidak ditemukan!")
        return

    df     = pd.read_csv(files[-1], index_col="Date", parse_dates=True).dropna()
    latest = df.iloc[-1]

    payload = {
        "open"        : float(latest["Open"]),
        "high"        : float(latest["High"]),
        "low"         : float(latest["Low"]),
        "close"       : float(latest["Close"]),
        "volume"      : float(latest["Volume"]),
        "ma_7"        : float(latest["MA_7"]),
        "ma_14"       : float(latest["MA_14"]),
        "ma_30"       : float(latest["MA_30"]),
        "rsi_14"      : float(latest["RSI_14"]),
        "macd"        : float(latest["MACD"]),
        "signal"      : float(latest["Signal"]),
        "hist"        : float(latest["Hist"]),
        "bb_upper"    : float(latest["BB_upper"]),
        "bb_lower"    : float(latest["BB_lower"]),
        "daily_return": float(latest["Daily_Return"]),
        "close_lag1"  : float(latest["Close_lag1"]),
        "close_lag2"  : float(latest["Close_lag2"]),
        "close_lag5"  : float(latest["Close_lag5"]),
        "vol_ma_7"    : float(latest["Vol_MA_7"])
    }

    print(f"Input data tanggal: {df.index[-1].date()}")
    print(f"Close harga input : ${latest['Close']:.2f}")
    print()

    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code : {r.status_code}")
    print(f"Response    : {json.dumps(r.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  PENGUJIAN API — AAPL Stock Price Prediction")
    print("  Rafly Januar Raharjo | 235150201111011")
    print("="*55 + "\n")

    test_health()
    test_root()
    test_model_info()
    test_predict_latest()
    test_predict_manual()

    print("="*55)
    print("  SEMUA TEST SELESAI!")
    print("="*55)
