# 📈 MLOps-StockPricePrediction

> Sistem MLOps untuk prediksi harga saham harian **AAPL (Apple Inc.)** menggunakan **Random Forest Regressor** dengan strategi **Continuous Training**.

![Python](https://img.shields.io/badge/Python-3.11-blue)

**Mata Kuliah:** MLOps - Kelas B | Universitas Brawijaya 2025  
**Nama:** Rafly Januar Raharjo | **NIM:** 235150201111011

---

## 📌 Deskripsi Proyek

Proyek ini membangun sistem Machine Learning production-ready untuk prediksi harga penutupan saham AAPL (T+1) berbasis prinsip MLOps. Sistem dirancang dengan mekanisme **Continuous Training** untuk menjaga relevansi model terhadap perubahan pola pasar.

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Language | Python 3.11 |
| ML Model | scikit-learn (Random Forest) |
| Data Source | Yahoo Finance API (yfinance) |
| API Backend | FastAPI |
| Frontend | HTML + CSS + JavaScript |
| Experiment Tracking | MLflow |
| Drift Detection | Evidently AI |
| Dev Environment | GitHub Codespaces |

---

## 📁 Struktur Direktori
```
MLOps-StockPricePrediction/
├── .devcontainer/            # Konfigurasi GitHub Codespaces
│   └── devcontainer.json
├── .github/
│   └── workflows/            # CI/CD pipelines
├── config/                   # Konfigurasi sistem & hyperparameter
├── data/
│   ├── raw/                  # Data mentah dari Yahoo Finance
│   ├── processed/            # Data hasil feature engineering
│   └── external/             # Data eksternal (indeks makro)
├── docs/                     # Dokumentasi teknis
├── models/
│   ├── registry/             # Versioned model artifacts (.pkl)
│   └── experiments/          # Model dari eksperimen
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory Data Analysis AAPL
├── src/
│   ├── data/                 # Data ingestion & preprocessing
│   ├── features/             # Feature engineering (MA, RSI, MACD)
│   ├── models/               # Training & evaluasi Random Forest
│   ├── monitoring/           # Drift detection & retraining trigger
│   └── api/
│       ├── static/
│       │   ├── css/          # Styling web prediksi
│       │   └── js/           # Logic frontend & chart
│       ├── templates/        # HTML halaman prediksi
│       └── main.py           # FastAPI backend
├── tests/                    # Unit & integration tests
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Cara Menjalankan via GitHub Codespaces

1. Klik tombol **"Code"** (hijau) di halaman repo
2. Pilih tab **"Codespaces"**
3. Klik **"Create codespace on main"**
4. Tunggu setup otomatis ±2-3 menit
5. Semua dependensi terinstall otomatis via `requirements.txt`
6. Environment siap digunakan!

---

## 💻 Cara Menjalankan Lokal
```bash
# 1. Clone repositori
git clone https://github.com/RaflyJanuarRaharjo/MLOps-StockPricePrediction.git
cd MLOps-StockPricePrediction

# 2. Buat virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan API
uvicorn src.api.main:app --reload
```


## 📊 Kriteria Keberhasilan

| Metrik | Target |
|--------|--------|
| RMSE | Serendah mungkin, konsisten antar periode |
| MAE | < 2% dari rata-rata harga AAPL |
| Uptime sistem | > 95% |
| Drift detection | Retraining otomatis jika RMSE naik > 10% |

---

## 🔄 Alur Sistem MLOps
```
Yahoo Finance API → Data Ingestion → Feature Engineering
       → Model Training (Random Forest) → Model Registry
       → REST API (FastAPI) → Web Dashboard (HTML/CSS)
       → Monitoring & Drift Detection → Auto Retraining
```

