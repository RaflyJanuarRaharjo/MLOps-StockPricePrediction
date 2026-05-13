

---

# 📈 MLOps-StockPricePrediction

End-to-end MLOps system for daily AAPL stock price prediction using **Random Forest Regressor** with continuous training and automated CI/CD pipelines.

**Course:** MLOps – Class B | Universitas Brawijaya 2026
**Author:** Rafly Januar Raharjo
**NIM:** 235150201111011

---

# 📌 Project Overview

This project implements a production-ready MLOps pipeline for predicting next-day AAPL (Apple Inc.) closing prices using machine learning and automated operational workflows.

### Key capabilities:

* Automated stock data ingestion from Yahoo Finance
* Feature engineering pipeline
* Data versioning using DVC
* Experiment tracking with MLflow
* Model registry management
* Continuous training workflow
* CI/CD automation using GitHub Actions
* FastAPI inference service
* Docker-based deployment

---

# 🏗️ System Architecture

```
Yahoo Finance API
        ↓
Data Ingestion Pipeline
        ↓
Raw Dataset (DVC Versioned)
        ↓
Preprocessing + Feature Engineering
        ↓
Training Pipeline
        ↓
MLflow Experiment Tracking
        ↓
Model Registry
        ↓
Production Model
        ↓
FastAPI Inference Service
```

---

# 🛠️ Tech Stack

| Component            | Technology              |
| -------------------- | ----------------------- |
| Programming Language | Python 3.11             |
| Machine Learning     | scikit-learn            |
| Model                | Random Forest Regressor |
| Data Source          | Yahoo Finance API       |
| API Framework        | FastAPI                 |
| Experiment Tracking  | MLflow 2.10.0           |
| Data Versioning      | DVC                     |
| Drift Detection      | Evidently AI            |
| Containerization     | Docker                  |
| CI/CD                | GitHub Actions          |
| Testing              | pytest                  |

---

# 📁 Project Structure

```
MLOps-StockPricePrediction/
├── .github/
│   └── workflows/
│       ├── daily_ingestion.yml
│       └── mlops-automation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   └── registry/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   ├── ingest_data.py
│   │   ├── preprocess.py
│   │   ├── pipeline.py
│   │   └── scheduler.py
│   ├── features/
│   │   └── feature_eng.py
│   └── models/
│       └── train.py
├── tests/
├── docker-compose.yaml
├── requirements.txt
├── inference.py
├── registry.py
└── README.md
```

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/RaflyJanuarRaharjo/MLOps-StockPricePrediction.git
cd MLOps-StockPricePrediction
```

## 2. Create Virtual Environment

```bash
py -3.11 -m venv venv311
```

Activate:

**Windows**

```bash
venv311\Scripts\activate
```

**Linux/macOS**

```bash
source venv311/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Running the ETL Pipeline

## Data Ingestion

```bash
python src/data/ingest_data.py
```

## Preprocessing

```bash
python src/data/preprocess.py
```

---

# 🧪 Model Training

```bash
python src/models/train.py
```

### Pipeline behavior:

* Train multiple Random Forest configurations
* Log metrics to MLflow
* Register best-performing model
* Save artifacts to Model Registry

---

# 📊 MLflow Tracking

## Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Access:

```
http://127.0.0.1:5000
```

---

# 🐳 Docker Deployment

## Start Services

```bash
docker compose up --build
```

## Services

| Service   | URL                                            |
| --------- | ---------------------------------------------- |
| MLflow UI | [http://localhost:5000](http://localhost:5000) |
| FastAPI   | [http://localhost:8000](http://localhost:8000) |

---

# ⚡ FastAPI Health Check

```
GET /health
```

Example:

```
http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": false
}
```

---

# 🔄 CI/CD Pipeline

Every push to `main` triggers:

```
Push to GitHub
      ↓
GitHub Actions
      ↓
Automated Testing
      ↓
Automated Training
      ↓
Model Evaluation
      ↓
Model Registry Update
```

---

# 📦 DVC Data Versioning

## Track dataset

```bash
dvc add data/raw/aapl_raw.csv
git add data/raw/aapl_raw.csv.dvc
git commit -m "track dataset with DVC"
```

---

# 🧪 Unit Testing

```bash
pytest tests/ -v
```

---

# 📈 Features Used

| Category        | Features                           |
| --------------- | ---------------------------------- |
| OHLCV           | Open, High, Low, Close, Volume     |
| Moving Average  | MA_7, MA_14, MA_30                 |
| Momentum        | RSI_14, MACD                       |
| Volatility      | Bollinger Bands                    |
| Lag Features    | Close_lag1, Close_lag2, Close_lag5 |
| Volume Features | Vol_MA_7                           |

**Total Features:** 19

---

# 🏆 Model Performance

* RMSE: 4.585
* R²: 0.725
* MAPE: 1.32%

**Production Model:**

```
AAPL-RF-Production@production
```

---

# ⚠️ Python Compatibility Note

This project is optimized for:

```
Python 3.11
```

Known issues may occur with:

* numpy
* mlflow
* scikit-learn

when using Python 3.13+

---

# ✅ Implemented Components

| Module              | Status |
| ------------------- | ------ |
| ETL Pipeline        | ✅      |
| Feature Engineering | ✅      |
| MLflow Tracking     | ✅      |
| Model Registry      | ✅      |
| DVC Versioning      | ✅      |
| FastAPI Inference   | ✅      |
| Docker Deployment   | ✅      |
| CI/CD Automation    | ✅      |
| Continuous Training | ✅      |

---

# 👨‍💻 Author

**Rafly Januar Raharjo**
Informatics Engineering – Universitas Brawijaya

GitHub: [https://github.com/RaflyJanuarRaharjo](https://github.com/RaflyJanuarRaharjo)

---

* bikin versi **lebih “ATS / recruiter optimized” (LinkedIn-ready)**
* atau bikin **README + badge + diagram gambar (Mermaid/Draw.io style)** biar makin keren portfolio kamu 🔥
