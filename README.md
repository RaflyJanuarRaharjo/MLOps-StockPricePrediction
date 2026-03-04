# MLOps-StockPricePrediction

Sistem MLOps untuk prediksi harga saham harian **AAPL (Apple Inc.)** menggunakan **Random Forest Regressor** dengan strategi **Continuous Training**.

**Mata Kuliah:** MLOps - Kelas B | Universitas Brawijaya 2025

---

## Tech Stack
- Python 3.11 | scikit-learn | pandas | numpy
- FastAPI (REST API serving)
- MLflow (experiment tracking & model registry)
- Evidently AI (drift detection)
- GitHub Codespaces (reproducible dev environment)
- Yahoo Finance API (data source)

---

## Struktur Direktori
```
MLOps-StockPricePrediction/
├── .devcontainer/        # Konfigurasi GitHub Codespaces
├── .github/workflows/    # CI/CD pipelines
├── config/               # Konfigurasi sistem
├── data/
│   ├── raw/              # Data mentah dari Yahoo Finance
│   ├── processed/        # Data hasil feature engineering
│   └── external/         # Data eksternal
├── docs/                 # Dokumentasi teknis
├── models/
│   ├── registry/         # Versioned model artifacts
│   └── experiments/      # Model eksperimen
├── notebooks/            # Jupyter notebooks
├── src/
│   ├── data/             # Data ingestion
│   ├── features/         # Feature engineering
│   ├── models/           # Training & evaluasi
│   ├── monitoring/       # Drift detection
│   └── api/              # REST API (FastAPI)
└── tests/                # Unit & integration tests
```

---

## Cara Menjalankan via GitHub Codespaces
1. Klik tombol **Code** (hijau) di halaman repo
2. Pilih tab **Codespaces**
3. Klik **Create codespace on main**
4. Tunggu setup otomatis ~2-3 menit
5. Environment siap digunakan!

## Cara Menjalankan Lokal
```bash
git clone https://github.com/RaflyJanuarRaharjo/MLOps-StockPricePrediction.git
cd MLOps-StockPricePrediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Branching Strategy (GitHub Flow)
- `main` - branch utama, selalu stable
- `feat/*` - branch untuk fitur baru / eksperimen
- `fix/*` - branch untuk perbaikan bug
- Setiap perubahan wajib melalui **Pull Request** sebelum merge ke main

---

## Kriteria Keberhasilan
| Metrik | Target |
|--------|--------|
| RMSE | Serendah mungkin, konsisten antar periode |
| MAE | < 2% dari rata-rata harga |
| Uptime sistem | > 95% |
| Drift detection | Retraining otomatis jika RMSE naik > 10% |
