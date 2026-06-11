# 📈 MLOps-StockPricePrediction

> Sistem MLOps end-to-end untuk prediksi harga saham harian **AAPL (Apple Inc.)** menggunakan **Random Forest Regressor** dengan strategi **Continuous Training**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.10.0-orange)
![DVC](https://img.shields.io/badge/DVC-enabled-green)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-automated-brightgreen)
![CI/CD](https://img.shields.io/badge/CI%2FCD-passing-success)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-orange)
![Grafana](https://img.shields.io/badge/Grafana-dashboard-yellow)

---

## 📚 Informasi Proyek

| Keterangan  | Detail                |
| ----------- | --------------------- |
| Mata Kuliah | MLOps — Kelas B       |
| Universitas | Universitas Brawijaya |
| Tahun       | 2026                  |
| Nama        | Rafly Januar Raharjo  |
| NIM         | 235150201111011       |

---

# 📌 Deskripsi Proyek

Proyek ini membangun sistem **Machine Learning production-ready** untuk memprediksi harga penutupan saham **AAPL (Apple Inc.)** pada hari berikutnya (**T+1**) menggunakan algoritma **Random Forest Regressor** berbasis prinsip **MLOps**.


---

# 🛠️ Tech Stack

| Komponen                | Teknologi                              |
| ----------------------- | -------------------------------------- |
| Language                | Python 3.11                            |
| ML Model                | scikit-learn (Random Forest Regressor) |
| Data Source             | Yahoo Finance API (yfinance)           |
| API Backend             | FastAPI                                |
| Frontend                | HTML, CSS, JavaScript                  |
| Experiment Tracking     | MLflow 2.10.0                          |
| Drift Detection         | Evidently AI                           |
| Data Versioning         | DVC                                    |
| CI/CD                   | GitHub Actions                         |
| Testing                 | pytest                                 |
| Containerization        | Docker + Docker Compose                |
| Load Balancer           | Nginx                                  |
| Metrics Scraping        | Prometheus                             |
| Monitoring Dashboard    | Grafana                                |
| Development Environment | GitHub Codespaces                      |

---

# 📁 Struktur Direktori

```bash
MLOps-StockPricePrediction/
├── .github/
│   └── workflows/
│       ├── daily_ingestion.yml
│       ├── mlops-automation.yaml
│       └── continuous_training.yml   # LK-12
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
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yaml
│       └── dashboards/
│           └── dashboard.yaml
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yaml
├── prometheus.yml
├── alert_rules.yml                   # LK-12
├── generate_load.py
├── ct_pipeline.py                    # LK-12
├── simulate_drift.py                 # LK-12
├── mlflow.db
├── registry.py
├── inference.py
└── README.md
```

---

# 🐳 Menjalankan Sistem dengan Docker Compose

## 📌 Prasyarat

Install Docker Desktop: https://www.docker.com/products/docker-desktop/

## ▶️ Menjalankan Sistem

```bash
docker compose up -d
```

## 📊 Mengecek Status Container

```bash
docker compose ps
```

Sistem menjalankan **7 layanan**:

| Container | Deskripsi |
|-----------|-----------|
| mlflow-server | MLflow Tracking Server |
| api-service (x3) | FastAPI Inferensi Model (3 replika) |
| load-balancer | Nginx Load Balancer |
| prometheus | Metrics Scraper |
| grafana | Monitoring Dashboard |

## 🌐 Akses Layanan

| Layanan        | URL                                          | Fungsi                          |
| -------------- | -------------------------------------------- | ------------------------------- |
| FastAPI        | http://localhost:8000/docs                   | Swagger UI inferensi model      |
| MLflow UI      | http://localhost:5000                        | Dashboard eksperimen            |
| API Health     | http://localhost:8000/health                 | Status API                      |
| Predict Latest | http://localhost:8000/predict-latest         | Prediksi terbaru                |
| Predict Manual | http://localhost:8000/predict                | Prediksi manual                 |
| Model Info     | http://localhost:8000/model-info             | Informasi model                 |
| Metrics        | http://localhost:8000/metrics                | Prometheus metrics endpoint     |
| Prometheus     | http://localhost:9090                        | Prometheus UI                   |
| Grafana        | http://localhost:3000                        | Monitoring Dashboard            |

## ⛔ Menghentikan Semua Layanan

```bash
docker compose down
```

---

# 🔌 Endpoint API

## 1️⃣ Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-05-20T07:14:28.664802"
}
```

## 2️⃣ Root Endpoint

```bash
GET /
```

## 3️⃣ Informasi Model

```bash
GET /model-info
```

## 4️⃣ Prediksi Otomatis

```bash
GET /predict-latest
```

Response:
```json
{
  "input_date": "2026-05-07",
  "prediction": 280.5,
  "message": "Berdasarkan data 2026-05-07, prediksi harga AAPL besok: $280.50",
  "model": "AAPL-RF-Production@production"
}
```

## 5️⃣ Prediksi Manual

```bash
POST /predict
Content-Type: application/json
```

Body:
```json
{
  "open": 285.0, "high": 290.0, "low": 280.0, "close": 287.44,
  "volume": 50000000, "ma_7": 283.0, "ma_14": 281.0, "ma_30": 278.0,
  "rsi_14": 55.0, "macd": 1.5, "signal": 1.2, "hist": 0.3,
  "bb_upper": 295.0, "bb_lower": 270.0, "daily_return": 0.005,
  "close_lag1": 285.0, "close_lag2": 283.0, "close_lag5": 279.0,
  "vol_ma_7": 48000000
}
```

## 6️⃣ Prometheus Metrics

```bash
GET /metrics
```

Metrik yang tersedia:
* `api_requests_total` — total request per endpoint
* `api_request_duration_seconds` — latensi inferensi
* `aapl_prediction_price_usd` — nilai prediksi harga AAPL
* `aapl_predictions_total` — total prediksi
* `system_cpu_usage_percent` — CPU usage
* `system_memory_usage_percent` — memory usage

---

# 📊 Monitoring Observabilitas (LK-11)

Stack monitoring **Prometheus + Grafana** memantau performa model dan infrastruktur secara real-time.

## Arsitektur Monitoring

```
API Service (/metrics)
      ↓
Prometheus (scrape setiap 10 detik)
      ↓
Grafana Dashboard (visualisasi real-time)
```

## Grafana Dashboard

Login: **http://localhost:3000** → `admin` / `admin123`

| Panel | Query | Fungsi |
|-------|-------|--------|
| Latensi Inferensi | `rate(api_request_duration_seconds_sum[5m]) / rate(api_request_duration_seconds_count[5m])` | Waktu respons |
| Throughput | `rate(api_requests_total[1m])` | Request per second |
| CPU & Memory | `system_cpu_usage_percent` | Utilisasi sumber daya |
| Prediksi Harga AAPL | `aapl_prediction_price_usd` | Deteksi drift |

## Simulasi Beban Kerja

```bash
python generate_load.py
```

---

# 🔄 Continuous Training Pipeline (LK-12)

Sistem CT menutup siklus MLOps dengan retraining otomatis saat terdeteksi perubahan data atau penurunan performa model.

## Arsitektur CT Pipeline

```
Trigger (Data/Schedule/Manual)
        ↓
GitHub Actions: continuous_training.yml
        ↓
Step 1: Deteksi Trigger & Validasi Data
        ↓
Step 2: Persiapan Data Terbaru
        ↓
Step 3: Continuous Training + Evaluasi Komparatif
        ↓
Step 4: Commit Data & Result ke Repo
```

## Skenario Trigger

| Skenario | Jenis | Kondisi | Mekanisme |
|----------|-------|---------|-----------|
| A — Performance-based | Otomatis | Prediksi < $200 atau > $400, latensi > 2s | Prometheus Alert Rules |
| B — Data-based | Otomatis | File baru di `data/processed/**` | GitHub Actions path trigger |
| C — Schedule-based | Otomatis | Setiap Minggu pukul 05:00 WIB | Cron job GitHub Actions |
| D — Manual | Manual | Dipicu via GitHub Actions UI | workflow_dispatch |

## Threshold Validasi Model

Model baru hanya dipromosikan ke `@production` jika memenuhi **semua** threshold:

| Metrik | Threshold | Keterangan |
|--------|-----------|------------|
| RMSE | ≤ 15.0 | Error rata-rata maksimum $15 |
| MAE | ≤ 10.0 | Absolut error maksimum $10 |
| R² | ≥ 0.50 | Minimal 50% variansi dijelaskan |
| MAPE | ≤ 5.0% | Error persentase maksimum 5% |

Model juga harus **lebih baik dari model production sebelumnya**.

## Menjalankan CT Pipeline

```bash
# Manual via terminal
python ct_pipeline.py manual

# Simulasi data drift
python simulate_drift.py
git add data/processed/
git commit -m "data: add shifted data for CT simulation"
git push origin main

# Cek hasil
cat ct_result.json
```

## Hasil Simulasi

Data di-shift +24.9% (harga $249–$360 vs asli $199–$287):

| Metrik | Nilai | Status |
|--------|-------|--------|
| RMSE | 5.617 | ✅ ≤ 15.0 |
| MAE | 4.3744 | ✅ ≤ 10.0 |
| R² | 0.8506 | ✅ ≥ 0.50 |
| MAPE | 1.3177% | ✅ ≤ 5.0% |
| Dipromosikan | true | ✅ |

Pipeline berjalan **tanpa intervensi manual** dalam **1 menit 50 detik**.

---

# ⚖️ Horizontal Scaling

```bash
# Tambah replika
docker compose up -d --scale api-service=5

# Kurangi replika
docker compose up -d --scale api-service=1
```

---

# 🚀 Menjalankan Lokal Tanpa Docker

```bash
git clone https://github.com/RaflyJanuarRaharjo/MLOps-StockPricePrediction.git
cd MLOps-StockPricePrediction
pip install yfinance pandas numpy mlflow scikit-learn dvc pytest fastapi uvicorn prometheus-client psutil
git pull origin main
python src/data/ingest_data.py
python src/data/preprocess.py
python src/models/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
python inference.py
```

---

# 🔄 Mengecek Prediksi AAPL Terbaru

```bash
git pull origin main
python src/data/ingest_data.py
python src/data/preprocess.py
python inference.py
```

> Data tersedia setelah NYSE tutup pukul 16:00 ET (04:00 WIB). Prediksi pada tanggal T menghasilkan harga penutupan T+1.

---

# 📦 Data Versioning dengan DVC

```bash
dvc add data/raw/aapl_raw_20260331_170609.csv
git add data/raw/aapl_raw_20260331_170609.csv.dvc
git commit -m "data(v1.0.0): track initial dataset"
git tag data-v1.0.0
dvc diff HEAD~1 HEAD
```

---

# 🧪 MLflow Experiment Tracking

| Run Name           | n_estimators | max_depth | RMSE   | R²     |
| ------------------ | ------------ | --------- | ------ | ------ |
| RF-Baseline        | 100          | None      | 4.4145 | 0.8549 |
| RF-Deep-Trees      | 200          | 20        | 4.5028 | 0.8490 |
| RF-Shallow-Trees ⭐ | 350          | 10        | 4.3644 | 0.8582 |

---

# 🏆 Model Production

**Model:** `AAPL-RF-Production` — **Version 7** — **Alias: @production**

| Metrik | Nilai   | Keterangan                  |
| ------ | ------- | --------------------------- |
| RMSE   | 4.3644  | Error rata-rata ±$4.36      |
| MAE    | 3.3308  | Absolut error rata-rata     |
| R²     | 0.8582  | Menjelaskan 85.82% variansi |
| MAPE   | 1.2502% | Error persentase rendah     |

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/AAPL-RF-Production@production")
predictions = model.predict(X)
```

---

# ⚙️ CI/CD Pipeline — Code as Trigger

```text
git push origin main
        ↓
pytest (6 unit tests)
        ↓
Auto Training + MLflow Logging
        ↓
Validasi Threshold (RMSE<15, R²>0.5, MAPE<5%)
        ↓
Auto Registry → @staging
```

---

# ⏰ Automasi Harian

Pipeline berjalan otomatis setiap Senin–Jumat pukul 04:00 WIB:

```text
daily_ingestion.yml → ingest_data.py → preprocess.py → git push
```

---

# 📊 Fitur Teknikal


Model **Random Forest Regressor** memanfaatkan 19 fitur hasil pengolahan data historis saham Apple Inc. (AAPL). Fitur-fitur ini terdiri dari data harga dasar, indikator teknikal, serta hasil *feature engineering* untuk membantu model mengenali pola pergerakan harga saham.

| No. | Fitur | Deskripsi | Tujuan |
|------|--------|-----------|---------|
| 1 | **Open** | Harga pertama saat pasar NYSE dibuka pada hari perdagangan. | Menunjukkan ekspektasi awal investor terhadap harga saham pada hari tersebut. |
| 2 | **High** | Harga tertinggi yang dicapai selama satu hari perdagangan. | Menggambarkan tingkat optimisme pasar pada hari tersebut. |
| 3 | **Low** | Harga terendah yang dicapai selama satu hari perdagangan. | Menggambarkan tingkat pesimisme pasar pada hari tersebut. |
| 4 | **Close** | Harga terakhir saat pasar ditutup. | Menjadi acuan utama nilai saham dan salah satu fitur paling penting dalam prediksi. |
| 5 | **Volume** | Jumlah saham yang diperdagangkan dalam satu hari. | Mengukur aktivitas perdagangan dan minat investor. |
| 6 | **MA_7** | Rata-rata harga penutupan selama 7 hari terakhir. | Mengidentifikasi tren jangka pendek dan mengurangi fluktuasi harian. |
| 7 | **MA_14** | Rata-rata harga penutupan selama 14 hari terakhir. | Mengidentifikasi tren jangka menengah dengan pergerakan yang lebih stabil. |
| 8 | **MA_30** | Rata-rata harga penutupan selama 30 hari terakhir. | Menggambarkan arah tren jangka panjang. |
| 9 | **RSI_14** | Relative Strength Index dengan periode 14 hari (nilai 0–100). | Mengukur momentum harga dan mendeteksi kondisi overbought maupun oversold. |
| 10 | **MACD** | Selisih antara EMA 12 hari dan EMA 26 hari. | Mengidentifikasi arah tren dan perubahan momentum harga. |
| 11 | **Signal** | EMA 9 hari dari nilai MACD. | Memberikan sinyal konfirmasi untuk perubahan tren. |
| 12 | **Hist** | Selisih antara MACD dan Signal Line. | Mengukur kekuatan momentum naik atau turun. |
| 13 | **BB_upper** | Batas atas Bollinger Bands. | Menunjukkan area harga yang relatif tinggi berdasarkan volatilitas. |
| 14 | **BB_lower** | Batas bawah Bollinger Bands. | Menunjukkan area harga yang relatif rendah berdasarkan volatilitas. |
| 15 | **Daily_Return** | Persentase perubahan harga penutupan dibanding hari sebelumnya. | Mengukur keuntungan atau kerugian harian saham. |
| 16 | **Close_lag1** | Harga penutupan satu hari sebelumnya. | Memberikan informasi historis jangka sangat pendek untuk prediksi berikutnya. |
| 17 | **Close_lag2** | Harga penutupan dua hari sebelumnya. | Membantu model mengenali pola pergerakan dalam beberapa hari terakhir. |
| 18 | **Close_lag5** | Harga penutupan lima hari sebelumnya. | Menangkap pola mingguan pada pergerakan harga saham. |
| 19 | **Vol_MA_7** | Rata-rata volume perdagangan selama 7 hari terakhir. | Mengonfirmasi kekuatan tren berdasarkan aktivitas transaksi. |

---

# 🧪 Unit Testing

```bash
pytest tests/test_pipeline.py -v
```

---

# ✅ Status Implementasi

| LK    | Komponen                              | Status |
| ----- | ------------------------------------- | ------ |
| LK-2  | GitHub Repository + Codespaces        | ✅      |
| LK-3  | Rancangan Pipeline ETL + Diagram      | ✅      |
| LK-4  | ingest_data.py + preprocess.py        | ✅      |
| LK-5  | DVC Data Versioning                   | ✅      |
| LK-6  | MLflow Experiment Tracking            | ✅      |
| LK-7  | Model Registry + Inferensi Production | ✅      |
| LK-8  | CI/CD Code as Trigger                 | ✅      |
| LK-9  | Docker Compose Orchestration          | ✅      |
| LK-10 | Horizontal Scaling + Load Balancer    | ✅      |
| LK-11 | Monitoring Prometheus + Grafana       | ✅      |
| LK-12 | Continuous Training Pipeline          | ✅      |
| Bonus | GitHub Actions Automasi Harian        | ✅      |
