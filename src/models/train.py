# src/models/train.py
# ============================================================
# Model Training dengan MLflow Integration
# Random Forest Regressor - AAPL Stock Price Prediction
# Rafly Januar Raharjo - 235150201111011 | MLOps Kelas B
# ============================================================

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

# --- Path ---
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "registry")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Fitur sesuai LK-3 ---
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_7", "MA_14", "MA_30",
    "RSI_14",
    "MACD", "Signal", "Hist",
    "BB_upper", "BB_lower",
    "Daily_Return",
    "Close_lag1", "Close_lag2", "Close_lag5",
    "Vol_MA_7"
]
TARGET_COL = "Target"


def load_processed_data() -> pd.DataFrame:
    """Baca file processed terbaru dari data/processed/."""
    files = sorted(glob.glob(os.path.join(DATA_PROC, "aapl_features_*.csv")))
    if not files:
        raise FileNotFoundError(f"Tidak ada file processed di {DATA_PROC}")
    latest = files[-1]
    print(f"[INFO] Membaca data: {latest}")
    df = pd.read_csv(latest, index_col="Date", parse_dates=True)
    print(f"[INFO] Shape data  : {df.shape}")
    return df


def prepare_features(df: pd.DataFrame):
    """Siapkan fitur X dan target y."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Kolom tidak ditemukan: {missing}")

    df = df.dropna(subset=available + [TARGET_COL])
    X  = df[available].values
    y  = df[TARGET_COL].values
    print(f"[INFO] Fitur digunakan : {len(available)} kolom")
    print(f"[INFO] Jumlah sampel   : {len(X)}")
    return X, y, available


def train_and_evaluate(X, y, params: dict) -> dict:
    """
    Train Random Forest dengan TimeSeriesSplit 80:20.
    Return dict berisi model + semua metrik.
    """
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = RandomForestRegressor(
        n_estimators   = params["n_estimators"],
        max_depth      = params["max_depth"],
        min_samples_split = params["min_samples_split"],
        min_samples_leaf  = params["min_samples_leaf"],
        max_features   = params["max_features"],
        random_state   = 42,
        n_jobs         = -1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return {
        "model"       : model,
        "rmse"        : round(rmse, 4),
        "mae"         : round(mae, 4),
        "r2"          : round(r2, 4),
        "mape"        : round(mape, 4),
        "train_size"  : len(X_train),
        "test_size"   : len(X_test),
    }


def run_experiment(experiment_name: str = "AAPL-RandomForest"):
    """
    Jalankan 3 variasi eksperimen dengan parameter berbeda.
    Catat semua ke MLflow.
    """

    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # Load data
    df         = load_processed_data()
    X, y, features = prepare_features(df)

    # ========================================================
    # 3 Variasi Eksperimen (hyperparameter berbeda)
    # ========================================================
    experiments = [
        {
            "run_name"        : "RF-Baseline",
            "n_estimators"    : 100,
            "max_depth"       : None,
            "min_samples_split": 2,
            "min_samples_leaf" : 1,
            "max_features"    : "sqrt",
            "description"     : "Baseline - parameter default Random Forest"
        },
        {
            "run_name"        : "RF-Deep-Trees",
            "n_estimators"    : 200,
            "max_depth"       : 20,
            "min_samples_split": 5,
            "min_samples_leaf" : 2,
            "max_features"    : "sqrt",
            "description"     : "More estimators, limited depth to reduce overfitting"
        },
        {
            "run_name"        : "RF-Shallow-Trees",
            "n_estimators"    : 300,
            "max_depth"       : 10,
            "min_samples_split": 10,
            "min_samples_leaf" : 4,
            "max_features"    : 0.7,
            "description"     : "Shallow trees, more regularization"
        },
    ]

    results = []
    best_rmse  = float("inf")
    best_run   = None

    print("\n" + "="*60)
    print("  MLFLOW EXPERIMENT — AAPL Stock Price Prediction")
    print("  Rafly Januar Raharjo | 235150201111011 | MLOps Kelas B")
    print("="*60)

    for i, exp in enumerate(experiments, 1):
        run_name = exp["run_name"]
        params   = {k: v for k, v in exp.items()
                    if k not in ["run_name", "description"]}

        print(f"\n[RUN {i}/3] {run_name}")
        print(f"  Params: {params}")

        with mlflow.start_run(run_name=run_name):

            # --- Log Parameters ---
            mlflow.log_param("run_name",          run_name)
            mlflow.log_param("n_estimators",      params["n_estimators"])
            mlflow.log_param("max_depth",         str(params["max_depth"]))
            mlflow.log_param("min_samples_split", params["min_samples_split"])
            mlflow.log_param("min_samples_leaf",  params["min_samples_leaf"])
            mlflow.log_param("max_features",      str(params["max_features"]))
            mlflow.log_param("description",       exp["description"])
            mlflow.log_param("ticker",            "AAPL")
            mlflow.log_param("n_features",        len(features))
            mlflow.log_param("train_test_split",  "80:20 TimeSeriesSplit")

            # --- Train & Evaluate ---
            result = train_and_evaluate(X, y, params)

            # --- Log Metrics ---
            mlflow.log_metric("rmse",       result["rmse"])
            mlflow.log_metric("mae",        result["mae"])
            mlflow.log_metric("r2_score",   result["r2"])
            mlflow.log_metric("mape",       result["mape"])
            mlflow.log_metric("train_size", result["train_size"])
            mlflow.log_metric("test_size",  result["test_size"])

            # --- Log Model ---
            signature = mlflow.models.infer_signature(X, result["model"].predict(X))
            mlflow.sklearn.log_model(
                sk_model       = result["model"],
                artifact_path  = "random_forest_model",
                signature      = signature,
                registered_model_name = f"AAPL-RF-{run_name}"
            )

            # --- Log Feature Importance ---
            fi = result["model"].feature_importances_
            fi_dict = {features[j]: round(float(fi[j]), 4) for j in range(len(features))}
            mlflow.log_dict(fi_dict, "feature_importance.json")

            run_id = mlflow.active_run().info.run_id

        print(f"  RMSE  : {result['rmse']}")
        print(f"  MAE   : {result['mae']}")
        print(f"  R²    : {result['r2']}")
        print(f"  MAPE  : {result['mape']}%")
        print(f"  Run ID: {run_id}")

        results.append({
            "run_name" : run_name,
            "run_id"   : run_id,
            "params"   : params,
            **{k: v for k, v in result.items() if k != "model"}
        })

        # Cek model terbaik
        if result["rmse"] < best_rmse:
            best_rmse = result["rmse"]
            best_run  = results[-1]
            best_model = result["model"]

    # ========================================================
    # SUMMARY
    # ========================================================
    print("\n" + "="*60)
    print("  HASIL SEMUA EKSPERIMEN")
    print("="*60)
    print(f"{'Run Name':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['run_name']:<22} {r['rmse']:>8} {r['mae']:>8} {r['r2']:>8} {r['mape']:>7}%")

    print("\n" + "="*60)
    print(f"  MODEL TERBAIK : {best_run['run_name']}")
    print(f"  RMSE terbaik  : {best_run['rmse']}")
    print(f"  MAE terbaik   : {best_run['mae']}")
    print(f"  R² terbaik    : {best_run['r2']}")
    print(f"  MAPE terbaik  : {best_run['mape']}%")
    print(f"  Run ID        : {best_run['run_id']}")
    print("="*60)

    # Simpan model terbaik ke models/registry/
    import pickle
    date_str   = datetime.now().strftime("%Y%m%d")
    model_path = os.path.join(MODEL_DIR, f"rf_aapl_best_{date_str}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[INFO] Model terbaik disimpan: {model_path}")
    print("\n[INFO] Buka MLflow UI dengan perintah:")
    print("       mlflow ui")
    print("       Lalu buka browser: http://localhost:5000")
    print("="*60)

    return results, best_run


if __name__ == "__main__":
    results, best = run_experiment()
