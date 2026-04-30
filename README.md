# 📈 MLOps-StockPricePrediction

> Sistem MLOps end-to-end untuk prediksi harga saham harian **AAPL (Apple Inc.)** menggunakan **Random Forest Regressor** dengan strategi **Continuous Training**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.11.1-orange)
![DVC](https://img.shields.io/badge/DVC-enabled-green)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-automated-brightgreen)

**Mata Kuliah:** MLOps - Kelas B | Universitas Brawijaya 2026  
**Nama:** Rafly Januar Raharjo | **NIM:** 235150201111011

---

## 📌 Deskripsi Proyek

Proyek ini membangun sistem Machine Learning production-ready untuk prediksi harga penutupan saham AAPL (T+1) berbasis prinsip MLOps. Sistem mencakup pipeline data otomatis, versioning dataset, experiment tracking, model registry, dan inferensi production.

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Language | Python 3.11 |
| ML Model | scikit-learn (Random Forest Regressor) |
| Data Source | Yahoo Finance API (yfinance) |
| API Backend | FastAPI |
| Frontend | HTML + CSS + JavaScript |
| Experiment Tracking | MLflow 3.11.1 |
| Drift Detection | Evidently AI |
| Data Versioning | DVC |
| Automation | GitHub Actions |
| Dev Environment | GitHub Codespaces |

---

## 📁 Struktur Direktori

```
MLOps-StockPricePrediction/
├── .github/
│   └── workflows/
│       └── daily_ingestion.yml   # Automasi harian GitHub Actions
├── .dvc/                         # Konfigurasi DVC
├── data/
│   ├── raw/                      # Data mentah dari Yahoo Finance
│   │   ├── aapl_raw_*.csv        # File data OHLCV
│   │   └── aapl_raw_*.csv.dvc    # DVC metadata
│   ├── processed/                # Data hasil feature engineering
│   │   └── aapl_features_*_v1.0.0.csv
│   └── external/                 # Data eksternal (indeks makro)
├── models/
│   └── registry/                 # Model artifacts (.pkl)
├── src/
│   ├── data/
│   │   ├── ingest_data.py        # Script pengambilan data
│   │   ├── preprocess.py         # Script preprocessing
│   │   ├── pipeline.py           # Main ETL runner
│   │   └── scheduler.py          # Scheduler harian
│   ├── features/
│   │   └── feature_eng.py        # Feature engineering
│   └── models/
│       └── train.py              # Training + MLflow logging
├── mlflow.db                     # MLflow SQLite backend
├── registry.py                   # Model registry management
├── inference.py                  # Verifikasi inferensi
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repositori
```bash
git clone https://github.com/RaflyJanuarRaharjo/MLOps-StockPricePrediction.git
cd MLOps-StockPricePrediction
```

### 2. Install Dependencies
```bash
pip install yfinance pandas numpy mlflow scikit-learn dvc
```

### 3. Jalankan ETL Pipeline
```bash
python src/data/ingest_data.py
python src/data/preprocess.py
```

### 4. Training Model dengan MLflow
```bash
python src/models/train.py
```

### 5. Buka MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Buka browser: http://127.0.0.1:5000
```

### 6. Inferensi Model Production
```bash
python inference.py
```

---

## 🔄 Pipeline Data (ETL)

```
Yahoo Finance API
      ↓ (ingest_data.py)
data/raw/aapl_raw_{timestamp}.csv
      ↓ (preprocess.py)
data/processed/aapl_features_{timestamp}_v1.0.0.csv
      ↓ (train.py)
MLflow Experiment → Model Registry → Production
```

---

## 📦 Data Versioning dengan DVC

### Track dataset awal
```bash
dvc add data/raw/aapl_raw_20260331_170609.csv
git add data/raw/aapl_raw_20260331_170609.csv.dvc
git commit -m "data(v1.0.0): track initial dataset"
git tag data-v1.0.0
```

### Update dataset (Continual Learning)
```bash
python src/data/ingest_data.py
dvc add data/raw/aapl_raw_20260331_170609.csv
git add data/raw/aapl_raw_20260331_170609.csv.dvc
git commit -m "data(v1.1.0): update dataset"
git tag data-v1.1.0
```

### Lihat perubahan antar versi
```bash
dvc diff HEAD~1 HEAD
```

---

## 🧪 MLflow Experiment Tracking

| Run Name | n_estimators | max_depth | RMSE | R² |
|----------|-------------|-----------|------|-----|
| RF-Baseline | 100 | None | 4.8743 | 0.6901 |
| RF-Deep-Trees | 200 | 20 | 4.8370 | 0.6948 |
| **RF-Shallow-Trees** ⭐ | **300** | **10** | **4.5852** | **0.7257** |

---

## 🏆 Model Registry & Versi Aktif

| Model Name | Version | Stage | RMSE |
|------------|---------|-------|------|
| AAPL-RF-RF-Baseline | v1 | None | 4.8743 |
| AAPL-RF-RF-Deep-Trees | v1 | None | 4.8370 |
| AAPL-RF-RF-Shallow-Trees | v1 | None | 4.5852 |
| **AAPL-RF-Production** | **v1** | **Production** ✅ | **4.6109** |

### ✅ Model Aktif untuk Inferensi

**Model:** `AAPL-RF-Production` — **Version 1** — **Alias: @production**

**Alasan dipilih:**
- RMSE = 4.6109 (error prediksi rata-rata ±$4.61)
- R² = 0.7227 (menjelaskan 72% variansi harga AAPL)
- MAPE = 1.3253% (error persentase sangat rendah)
- Parameter: n_estimators=500, max_depth=8, min_samples_split=12

### Load model Production
```python
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.pyfunc.load_model('models:/AAPL-RF-Production@production')
predictions = model.predict(X)
```

---

## ⚙️ Automasi Harian (GitHub Actions)

Pipeline data berjalan otomatis setiap hari kerja **pukul 04:00 WIB**:

```
Setiap Senin–Jumat 04:00 WIB
        ↓
GitHub Actions (daily_ingestion.yml)
        ↓
python src/data/ingest_data.py
        ↓
python src/data/preprocess.py
        ↓
git commit + push ke main
```

Jalankan manual: tab **Actions** di GitHub → **"Daily Data Ingestion"** → **"Run workflow"**

---

## 📊 Fitur Teknikal (19 Fitur)

| Kategori | Fitur |
|----------|-------|
| OHLCV | Open, High, Low, Close, Volume |
| Moving Average | MA_7, MA_14, MA_30 |
| Momentum | RSI_14, MACD, Signal, Hist |
| Volatilitas | BB_upper, BB_lower |
| Return | Daily_Return |
| Lag | Close_lag1, Close_lag2, Close_lag5 |
| Volume | Vol_MA_7 |
| **Target** | **Close T+1** |

---

## ✅ Status Implementasi

| LK | Komponen | Status |
|----|----------|--------|
| LK-2 | GitHub Repository + Codespaces | ✅ |
| LK-3 | Rancangan Pipeline ETL + Diagram | ✅ |
| LK-4 | ingest_data.py + preprocess.py | ✅ |
| LK-5 | DVC Data Versioning | ✅ |
| LK-6 | MLflow Experiment Tracking | ✅ |
| LK-7 | Model Registry + Inferensi Production | ✅ |
| Bonus | GitHub Actions Automasi Harian | ✅ |
