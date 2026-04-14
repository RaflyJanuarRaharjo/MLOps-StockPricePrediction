# 📈 MLOps-StockPricePrediction

> Sistem MLOps untuk prediksi harga saham harian **AAPL (Apple Inc.)** menggunakan **Random Forest Regressor** dengan strategi **Continuous Training** dan **Data Versioning (DVC)**.

![Python](https://img.shields.io/badge/Python-3.11-blue)

**Mata Kuliah:** MLOps - Kelas B | Universitas Brawijaya 2025
**Nama:** Rafly Januar Raharjo | **NIM:** 235150201111011

---

## 📌 Deskripsi Proyek

Proyek ini membangun sistem Machine Learning production-ready untuk prediksi harga penutupan saham AAPL (T+1) berbasis prinsip MLOps. Sistem dirancang dengan mekanisme:

* **Continuous Training** untuk adaptasi terhadap perubahan pasar
* **Data Versioning menggunakan DVC** untuk melacak perubahan dataset secara efisien

---

## 🛠️ Tech Stack

| Komponen            | Teknologi                    |
| ------------------- | ---------------------------- |
| Language            | Python 3.11                  |
| ML Model            | scikit-learn (Random Forest) |
| Data Source         | Yahoo Finance API (yfinance) |
| API Backend         | FastAPI                      |
| Frontend            | HTML + CSS + JavaScript      |
| Experiment Tracking | MLflow                       |
| Drift Detection     | Evidently AI                 |
| Data Versioning     | DVC                          |
| Dev Environment     | GitHub Codespaces            |

---

## 📁 Struktur Direktori

```
MLOps-StockPricePrediction/
├── data/
│   └── raw/
│       ├── aapl_raw_20260331_170609.csv
│       ├── aapl_raw_20260331_170609.csv.dvc
│       └── .gitignore
├── src/
│   └── data/
│       └── ingest_data.py
├── .dvc/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Inisialisasi DVC

```bash
dvc init
git add .dvc .gitignore
git commit -m "init: initialize DVC"
```

---

## 📦 Data Versioning dengan DVC

### 🔹 Versi Awal Dataset (v1.0.0)

```bash
python src/data/ingest_data.py

dvc add data/raw/aapl_raw_20260331_170609.csv
git add data/raw/aapl_raw_20260331_170609.csv.dvc data/raw/.gitignore
git commit -m "data(v1.0.0): track initial dataset"
```

---

### 🔄 Continual Learning (Update Dataset v1.1.0)

Data diperbarui dengan ingestion ulang:

```bash
python src/data/ingest_data.py
```

Karena file baru dihasilkan, dilakukan overwrite untuk menjaga versioning:

```bash
move data/raw/aapl_raw_20260414_*.csv data/raw/aapl_raw_20260331_170609.csv
```

Tracking ulang dengan DVC:

```bash
dvc add data/raw/aapl_raw_20260331_170609.csv
git add data/raw/aapl_raw_20260331_170609.csv.dvc
git commit -m "data(v1.1.0): update dataset"
```

---

## 🔍 Audit Perubahan Data

Melihat perbedaan antar versi dataset:

```bash
dvc diff
```

Output menunjukkan:

* perubahan hash dataset
* perubahan ukuran file

Hal ini membuktikan bahwa DVC melacak perubahan tanpa menyimpan file besar di Git.

---

## 💡 Konsep MLOps yang Diterapkan

* Data Versioning (DVC)
* Continuous Training
* Reproducibility
* Separation of Code & Data
* Monitoring & Drift Detection

---

## 🚀 Cara Menjalankan

### 💻 Lokal

```bash
git clone https://github.com/RaflyJanuarRaharjo/MLOps-StockPricePrediction.git
cd MLOps-StockPricePrediction

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

---

## ☁️ (Opsional) DVC Remote Storage

```bash
dvc remote add -d myremote ../dvc-storage
dvc push
```

---

## 📊 Alur Sistem MLOps

```
Yahoo Finance API → Data Ingestion → Data Versioning (DVC)
       → Feature Engineering → Model Training
       → Model Registry → API (FastAPI)
       → Monitoring → Retraining
```

---

## 📈 Insight

Dengan DVC, dataset besar tidak disimpan langsung di Git, melainkan sebagai metadata (.dvc).
Setiap perubahan data menghasilkan hash baru, sehingga histori dataset dapat dilacak dengan jelas dan reproducible.

---

## ✅ Kesimpulan

Proyek ini berhasil mengimplementasikan:

* Versioning dataset menggunakan DVC
* Simulasi continual learning
* Audit perubahan data antar versi

Sehingga mendukung praktik MLOps yang scalable, efisien, dan production-ready.
