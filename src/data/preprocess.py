import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

DATA_RAW  = "data/raw"
DATA_PROC = "data/processed"
os.makedirs(DATA_PROC, exist_ok=True)

def load_latest_raw():
    files = sorted(glob.glob(os.path.join(DATA_RAW, "aapl_raw_*.csv")))
    if not files:
        raise FileNotFoundError("Tidak ada file raw ditemukan!")
    latest = files[-1]
    print(f"[INFO] Membaca file: {latest}")
    df = pd.read_csv(latest, index_col="Date", parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def clean(df):
    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index.dayofweek < 5]
    df = df.ffill().bfill()
    Q1  = df["Close"].quantile(0.25)
    Q3  = df["Close"].quantile(0.75)
    IQR = Q3 - Q1
    df  = df[(df["Close"] >= Q1-1.5*IQR) & (df["Close"] <= Q3+1.5*IQR)]
    for col in ["Open","High","Low","Close"]:
        df[col] = df[col].astype(float)
    df = df.sort_index()
    print(f"[INFO] Cleaning: {before} -> {len(df)} baris")
    return df

def add_features(df):
    df["MA_7"]        = df["Close"].rolling(7).mean()
    df["MA_14"]       = df["Close"].rolling(14).mean()
    df["MA_30"]       = df["Close"].rolling(30).mean()
    delta             = df["Close"].diff()
    gain              = delta.where(delta>0,0).rolling(14).mean()
    loss              = (-delta.where(delta<0,0)).rolling(14).mean()
    df["RSI_14"]      = 100-(100/(1+gain/loss))
    ema12             = df["Close"].ewm(span=12,adjust=False).mean()
    ema26             = df["Close"].ewm(span=26,adjust=False).mean()
    df["MACD"]        = ema12-ema26
    df["Signal"]      = df["MACD"].ewm(span=9,adjust=False).mean()
    df["Hist"]        = df["MACD"]-df["Signal"]
    ma20              = df["Close"].rolling(20).mean()
    std20             = df["Close"].rolling(20).std()
    df["BB_upper"]    = ma20+2*std20
    df["BB_lower"]    = ma20-2*std20
    df["Daily_Return"]= df["Close"].pct_change()*100
    df["Close_lag1"]  = df["Close"].shift(1)
    df["Close_lag2"]  = df["Close"].shift(2)
    df["Close_lag5"]  = df["Close"].shift(5)
    df["Vol_MA_7"]    = df["Volume"].rolling(7).mean()
    df["Target"]      = df["Close"].shift(-1)
    before = len(df)
    df = df.dropna()
    print(f"[INFO] Features: {len(df.columns)} kolom, {len(df)} baris (hapus {before-len(df)} NaN)")
    return df

def preprocess():
    print("="*60)
    print("  PREPROCESS DATA - AAPL Stock Price Prediction")
    print("  Rafly Januar Raharjo | 235150201111011 | MLOps Kelas B")
    print(f"  Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    df = load_latest_raw()
    df = clean(df)
    df = add_features(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"aapl_features_{timestamp}_v1.0.0.csv"
    filepath  = os.path.join(DATA_PROC, filename)
    df.to_csv(filepath)

    print(f"\n[INFO] Shape akhir  : {df.shape}")
    print(f"[INFO] Semua kolom  : {list(df.columns)}")
    print(f"[INFO] File disimpan: {filepath}")
    print(f"\n[DATA] Sample 5 baris terakhir:")
    print(df[["Close","MA_7","RSI_14","MACD","BB_upper","Target"]].tail().round(2).to_string())
    print(f"\n[OK] PREPROCESS BERHASIL!")
    print("="*60)
    return filepath

if __name__ == "__main__":
    preprocess()
