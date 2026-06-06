# simulate_drift.py
# ============================================================
# Simulasi Data Drift - LK-12
# Membuat data dengan distribusi berbeda untuk memicu CT
# Rafly Januar Raharjo - 235150201111011 | MLOps Kelas B
# ============================================================

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta

DATA_PROC = "data/processed"
os.makedirs(DATA_PROC, exist_ok=True)

def simulate_drift():
    print("="*55)
    print("  SIMULASI DATA DRIFT - LK-12")
    print("="*55)

    # Load data terakhir sebagai baseline
    files = sorted(glob.glob(os.path.join(DATA_PROC, "aapl_features_*.csv")))
    if not files:
        raise FileNotFoundError("Tidak ada data processed!")

    df = pd.read_csv(files[-1], index_col="Date", parse_dates=True)
    print(f"[INFO] Base data: {files[-1]}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Close range asli: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    # ========================================================
    # Buat data shifted: harga naik 20-30% (simulasi drift)
    # ========================================================
    df_drift = df.copy()

    # Shift semua kolom harga ke atas (simulasi harga naik drastis)
    price_cols = ["Open", "High", "Low", "Close",
                  "MA_7", "MA_14", "MA_30",
                  "BB_upper", "BB_lower",
                  "Close_lag1", "Close_lag2", "Close_lag5", "Target"]

    drift_factor = np.random.uniform(1.20, 1.30)  # naik 20-30%
    print(f"[INFO] Drift factor: {drift_factor:.4f} (harga naik ~{(drift_factor-1)*100:.1f}%)")

    for col in price_cols:
        if col in df_drift.columns:
            df_drift[col] = df_drift[col] * drift_factor

    # Tambah noise kecil agar lebih realistis
    for col in price_cols:
        if col in df_drift.columns:
            noise = np.random.normal(0, df_drift[col].std() * 0.02, len(df_drift))
            df_drift[col] = df_drift[col] + noise

    # Update tanggal (extend ke depan)
    last_date = df_drift.index[-1]
    new_dates  = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=len(df_drift),
        freq='B'  # Business days
    )
    df_drift.index = new_dates
    df_drift.index.name = "Date"

    print(f"[INFO] Close range drift: ${df_drift['Close'].min():.2f} - ${df_drift['Close'].max():.2f}")

    # Simpan
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"aapl_features_{timestamp}_drift_v1.0.0.csv"
    filepath   = os.path.join(DATA_PROC, filename)
    df_drift.to_csv(filepath)

    print(f"[OK] Data drift disimpan: {filepath}")
    print(f"[INFO] Push ke GitHub untuk trigger CT pipeline!")
    print("="*55)

    return filepath


if __name__ == "__main__":
    simulate_drift()
