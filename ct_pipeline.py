# ct_pipeline.py
# ============================================================
# Continuous Training Pipeline - LK-12
# Auto retrain + compare + promote model
# Rafly Januar Raharjo - 235150201111011 | MLOps Kelas B
# ============================================================

import os
import glob
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME  = "AAPL-RF-Production"
DATA_PROC   = "data/processed"
MODEL_DIR   = "models/registry"

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_7", "MA_14", "MA_30", "RSI_14",
    "MACD", "Signal", "Hist",
    "BB_upper", "BB_lower", "Daily_Return",
    "Close_lag1", "Close_lag2", "Close_lag5", "Vol_MA_7"
]

# Threshold untuk validasi model baru
THRESHOLDS = {
    "rmse_max" : 15.0,
    "mae_max"  : 10.0,
    "r2_min"   : 0.50,
    "mape_max" : 5.0
}

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# HELPER: Load Data
# ============================================================

def load_latest_data():
    files = sorted(glob.glob(os.path.join(DATA_PROC, "aapl_features_*.csv")))
    if not files:
        raise FileNotFoundError("Tidak ada data processed!")
    latest = files[-1]
    print(f"[INFO] Data: {latest}")
    df = pd.read_csv(latest, index_col="Date", parse_dates=True).dropna()
    X  = df[FEATURE_COLS].values
    y  = df["Target"].values
    return X, y, latest

# ============================================================
# HELPER: Get Current Production Metrics
# ============================================================

def get_production_metrics():
    """Ambil metrik model production saat ini dari MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, "production")
        run = client.get_run(model_version.run_id)
        metrics = run.data.metrics
        print(f"[INFO] Model production saat ini: version {model_version.version}")
        print(f"[INFO] Metrik production: RMSE={metrics.get('rmse','N/A')} R2={metrics.get('r2_score','N/A')}")
        return metrics, model_version.version
    except Exception as e:
        print(f"[WARN] Tidak bisa ambil metrik production: {e}")
        return None, None

# ============================================================
# HELPER: Train New Model
# ============================================================

def train_new_model(X, y, trigger_reason):
    """Latih model baru dan log ke MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("AAPL-CT-Pipeline")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    params = {
        "n_estimators"     : 350,
        "max_depth"        : 10,
        "min_samples_split": 10,
        "min_samples_leaf" : 4,
        "max_features"     : 0.7
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"CT-{trigger_reason}-{timestamp}") as run:
        mlflow.log_param("trigger_reason", trigger_reason)
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_param("data_size", len(X))
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        mae  = round(float(mean_absolute_error(y_test, y_pred)), 4)
        r2   = round(float(r2_score(y_test, y_pred)), 4)
        mape = round(float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100), 4)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape", mape)

        signature = mlflow.models.infer_signature(X, model.predict(X))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            signature=signature,
            registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id

    print(f"[INFO] Model baru: RMSE={rmse} MAE={mae} R2={r2} MAPE={mape}%")
    print(f"[INFO] Run ID: {run_id}")

    return {
        "run_id": run_id,
        "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
        "model": model
    }

# ============================================================
# HELPER: Compare & Promote
# ============================================================

def compare_and_promote(new_metrics, old_metrics, new_run_id):
    """Bandingkan model baru vs lama, promote jika lebih baik."""
    client = mlflow.MlflowClient()

    print("\n" + "="*55)
    print("  EVALUASI KOMPARATIF")
    print("="*55)

    if old_metrics:
        old_rmse = old_metrics.get("rmse", 999)
        old_r2   = old_metrics.get("r2_score", 0)
        print(f"  Model LAMA  : RMSE={old_rmse:.4f} | R²={old_r2:.4f}")
        print(f"  Model BARU  : RMSE={new_metrics['rmse']} | R²={new_metrics['r2']}")
        print("-"*55)

        is_better = (
            new_metrics["rmse"] < old_rmse and
            new_metrics["r2"]   > old_r2
        )
    else:
        print("  Tidak ada model production sebelumnya.")
        is_better = True

    # Validasi threshold
    passes_threshold = (
        new_metrics["rmse"] <= THRESHOLDS["rmse_max"] and
        new_metrics["mae"]  <= THRESHOLDS["mae_max"]  and
        new_metrics["r2"]   >= THRESHOLDS["r2_min"]   and
        new_metrics["mape"] <= THRESHOLDS["mape_max"]
    )

    if is_better and passes_threshold:
        # Ambil versi terbaru dan promote
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest   = sorted(versions, key=lambda v: int(v.version))[-1]
        client.set_registered_model_alias(MODEL_NAME, "production", latest.version)
        client.set_registered_model_alias(MODEL_NAME, "staging", latest.version)
        print(f"  ✅ Model BARU dipromosikan ke @production (version {latest.version})")
        promoted = True
    else:
        if not passes_threshold:
            print(f"  ❌ Model BARU tidak memenuhi threshold, tidak dipromosikan")
        else:
            print(f"  ❌ Model BARU tidak lebih baik dari production, tidak dipromosikan")
        promoted = False

    print("="*55)
    return promoted

# ============================================================
# MAIN: Continuous Training Pipeline
# ============================================================

def run_ct_pipeline(trigger_reason="manual"):
    print("\n" + "="*55)
    print("  CONTINUOUS TRAINING PIPELINE - LK-12")
    print(f"  Trigger: {trigger_reason}")
    print(f"  Waktu  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    # 1. Load data terbaru
    X, y, data_file = load_latest_data()

    # 2. Ambil metrik model production saat ini
    old_metrics, old_version = get_production_metrics()

    # 3. Train model baru
    print("\n[STEP] Training model baru...")
    new_result = train_new_model(X, y, trigger_reason)

    # 4. Compare dan promote
    print("\n[STEP] Membandingkan model...")
    promoted = compare_and_promote(new_result, old_metrics, new_result["run_id"])

    # 5. Simpan hasil ke JSON
    result_summary = {
        "timestamp"      : datetime.now().isoformat(),
        "trigger_reason" : trigger_reason,
        "data_file"      : data_file,
        "new_model"      : {
            "run_id": new_result["run_id"],
            "rmse"  : new_result["rmse"],
            "mae"   : new_result["mae"],
            "r2"    : new_result["r2"],
            "mape"  : new_result["mape"]
        },
        "old_model": {
            "rmse": old_metrics.get("rmse") if old_metrics else None,
            "r2"  : old_metrics.get("r2_score") if old_metrics else None,
        },
        "promoted": promoted
    }

    with open("ct_result.json", "w") as f:
        json.dump(result_summary, f, indent=2)

    print(f"\n[INFO] Hasil disimpan: ct_result.json")
    print(f"[INFO] Status: {'✅ Model dipromosikan!' if promoted else '❌ Model tidak dipromosikan'}")
    print("="*55)

    return result_summary


if __name__ == "__main__":
    import sys
    trigger = sys.argv[1] if len(sys.argv) > 1 else "manual"
    run_ct_pipeline(trigger_reason=trigger)
