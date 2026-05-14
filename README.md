# 📈 MLOps-StockPricePrediction

> Sistem MLOps end-to-end untuk prediksi harga saham harian **AAPL (Apple Inc.)** menggunakan **Random Forest Regressor** dengan strategi **Continuous Training**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.10.0-orange)
![DVC](https://img.shields.io/badge/DVC-enabled-green)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-automated-brightgreen)
![CI/CD](https://img.shields.io/badge/CI%2FCD-passing-success)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

**Mata Kuliah:** MLOps - Kelas B | Universitas Brawijaya 2026  
**Nama:** Rafly Januar Raharjo | **NIM:** 235150201111011

---

## 📌 Deskripsi Proyek

Proyek ini membangun sistem Machine Learning production-ready untuk prediksi harga penutupan saham AAPL (T+1) berbasis prinsip MLOps. Sistem mencakup:

- **Pipeline data otomatis** — ingestion harian dari Yahoo Finance API
- **Data versioning** — DVC untuk melacak perubahan dataset
- **Experiment tracking** — MLflow untuk mencatat semua eksperimen
- **Model registry** — pengelolaan siklus hidup model dari Staging ke Production
- **CI/CD automation** — GitHub Actions untuk "Code as Trigger"
- **Continuous Training** — retraining otomatis berbasis perubahan kode
- **Container orchestration** — Docker Compose untuk menjalankan seluruh sistem

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Language | Python 3.11 |
| ML Model | scikit-learn (Random Forest Regressor) |
| Data Source | Yahoo Finance API (yfinance) |
| API Backend | FastAPI |
| Frontend | HTML + CSS + JavaScript |
| Experiment Tracking | MLflow 2.10.0 |
| Drift Detection | Evidently AI |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |
| Testing | pytest |
| Containerization | Docker + Docker Compose |
| Dev Environment | GitHub Codespaces |

---

## 📁 Struktur Direktori
```
MLOps-StockPricePrediction/
├── .github/
│   └── workflows/
│       ├── daily_ingestion.yml
│       └── mlops-automation.yaml
├── .dvc/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   └── registry/
├── src/
│   ├── data/
│   │   ├── ingest_data.py
│   │   ├── preprocess.py
│   │   ├── pipeline.py
│   │   └── scheduler.py
│   ├── features/
│   │   └── feature_eng.py
│   ├── models/
│   │   └── train.py
│   └── api/
│       └── main.py
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yaml
├── mlflow.db
├── registry.py
├── inference.py
└── README.md
```

## 🐳 Menjalankan Seluruh Sistem dengan Docker Compose

### Prasyarat
- Docker Desktop terinstall → https://www.docker.com/products/docker-desktop/

### Jalankan dengan 1 perintah:
```bash
docker compose up -d
```

### Cek status container:
```bash
docker compose ps
```

### Akses layanan:
| Layanan | URL | Fungsi |
|---------|-----|--------|
| **FastAPI** | http://localhost:8000/docs | Swagger UI inferensi model |
| **MLflow UI** | http://localhost:5000 | Dashboard eksperimen |
| **API Health** | http://localhost:8000/health | Status API |
| **Prediksi** | http://localhost:8000/predict-latest | Prediksi harga terbaru |

### Training model ke MLflow container:
```bash
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python src/models/train.py
```

### Hentikan semua layanan:
```bash
docker compose down
```

---

## 🚀 Cara Menjalankan Lokal (Tanpa Docker)

```bash
git clone https://github.com/RaflyJanuarRaharjo/MLOps-StockPricePrediction.git
cd MLOps-StockPricePrediction
pip install yfinance pandas numpy mlflow scikit-learn dvc pytest fastapi uvicorn
git pull origin main
python src/data/ingest_data.py
python src/data/preprocess.py
python src/models/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
python inference.py
```

---

## 🔄 Mengecek Prediksi Harga AAPL Terbaru

Jalankan setiap pagi setelah pukul 04:00 WIB:

```bash
git pull origin main
python src/data/ingest_data.py
python src/data/preprocess.py
python inference.py
```

> **Catatan:** Data tersedia setelah NYSE tutup pukul 16:00 ET (04:00 WIB). Prediksi pada tanggal T menghasilkan harga penutupan T+1.

---

## 📦 Data Versioning dengan DVC

```bash
dvc add data/raw/aapl_raw_20260331_170609.csv
git add data/raw/aapl_raw_20260331_170609.csv.dvc
git commit -m "data(v1.0.0): track initial dataset"
git tag data-v1.0.0
dvc diff HEAD~1 HEAD
```

---

## 🧪 MLflow Experiment Tracking

| Run Name | n_estimators | max_depth | RMSE | R² |
|----------|-------------|-----------|------|-----|
| RF-Baseline | 100 | None | 4.8743 | 0.6901 |
| RF-Deep-Trees | 200 | 20 | 4.8370 | 0.6948 |
| **RF-Shallow-Trees** ⭐ | **300** | **10** | **4.3644** | **0.8582** |

---

## 🏆 Model Aktif untuk Inferensi

**Model:** `AAPL-RF-Production` — **Version 1** — **Alias: @production**

| Metrik | Nilai | Keterangan |
|--------|-------|------------|
| RMSE | 4.3644 | Error rata-rata ±$4.36 |
| MAE | 3.3308 | Absolut error rata-rata |
| R² | 0.8582 | Menjelaskan 85.82% variansi |
| MAPE | 1.2502% | Error persentase sangat rendah |

```python
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.pyfunc.load_model('models:/AAPL-RF-Production@production')
predictions = model.predict(X)
```

---

## ⚙️ CI/CD Pipeline — Code as Trigger (LK-8)
git push origin main
↓
Tahap 1: pytest (6 unit tests)
↓
Tahap 2: Auto Training + MLflow logging
↓
Tahap 3: Validasi threshold (RMSE<10, R²>0.5, MAPE<5%)
↓
Tahap 4: Auto Registry → @staging

---

## ⏰ Automasi Harian

Pipeline data otomatis setiap **Senin–Jumat 04:00 WIB**:
daily_ingestion.yml → ingest_data.py → preprocess.py → git push

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

## 🧪 Unit Tests

```bash
pytest tests/test_pipeline.py -v
```

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
| LK-8 | CI/CD Code as Trigger | ✅ |
| LK-9 | Docker Compose Orchestration | ✅ |
| Bonus | GitHub Actions Automasi Harian | ✅ |
