import yfinance as yf
import pandas as pd
import os
from datetime import datetime

TICKER   = "AAPL"
DATA_RAW = "data/raw"
os.makedirs(DATA_RAW, exist_ok=True)

def ingest():
    print("="*60)
    print("  INGEST DATA - AAPL Stock Price Prediction")
    print("  Rafly Januar Raharjo | 235150201111011 | MLOps Kelas B")
    print(f"  Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\nMengambil data AAPL dari Yahoo Finance API...")
    df = yf.download(TICKER, period="1y", interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open","High","Low","Close","Volume"]]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    # Timestamp pada nama file agar tidak menimpa data lama
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"aapl_raw_{timestamp}.csv"
    filepath  = os.path.join(DATA_RAW, filename)
    df.to_csv(filepath)

    print(f"\n[INFO] Ticker          : {TICKER} (Apple Inc. - NASDAQ)")
    print(f"[INFO] Sumber          : Yahoo Finance API (yfinance)")
    print(f"[INFO] Total baris     : {len(df)} hari perdagangan")
    print(f"[INFO] Kolom           : {list(df.columns)}")
    print(f"[INFO] Rentang tanggal : {df.index[0].date()} sd {df.index[-1].date()}")
    print(f"[INFO] File disimpan   : {filepath}")
    print(f"\n[DATA] 5 Baris Pertama:")
    print(df.head().to_string())
    print(f"\n[DATA] 5 Baris Terakhir:")
    print(df.tail().to_string())
    print(f"\n[STATS] Statistik Deskriptif:")
    print(df.describe().round(2).to_string())
    print(f"\n[OK] INGEST BERHASIL! File: {filepath}")
    print("="*60)
    return filepath

if __name__ == "__main__":
    ingest()
