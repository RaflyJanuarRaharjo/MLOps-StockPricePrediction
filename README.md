# 🔄 Continuous Training Pipeline (LK-12)

Sistem **Continuous Training (CT)** menutup siklus MLOps dengan memastikan model selalu diperbarui secara otomatis ketika terdeteksi perubahan data atau penurunan performa.

## Arsitektur CT Pipeline

```
Trigger (Data/Schedule/Manual)
        ↓
GitHub Actions: continuous_training.yml
        ↓
Step 1: Deteksi Trigger & Validasi Data
        ↓
Step 2: Persiapan Data Terbaru (ingest + preprocess)
        ↓
Step 3: Continuous Training + Evaluasi Komparatif
        ↓
Step 4: Commit Data & Result ke Repo
```

## Skenario Trigger

| Skenario | Jenis | Kondisi | Mekanisme |
|----------|-------|---------|-----------|
| A — Performance-based | Otomatis | Prediksi < $200 atau > $400, latensi > 2s | Prometheus Alert Rules |
| B — Data-based | Otomatis | File baru masuk ke `data/processed/**` | GitHub Actions path trigger |
| C — Schedule-based | Otomatis | Setiap Minggu pukul 05:00 WIB | Cron job GitHub Actions |
| D — Manual | Manual | Dipicu langsung via GitHub Actions UI | workflow_dispatch |

## Threshold Validasi Model

Model baru hanya dipromosikan ke `@production` jika memenuhi **semua** threshold berikut:

| Metrik | Threshold | Keterangan |
|--------|-----------|------------|
| RMSE | ≤ 15.0 | Error rata-rata maksimum $15 |
| MAE | ≤ 10.0 | Absolut error maksimum $10 |
| R² | ≥ 0.50 | Minimal 50% variansi dijelaskan model |
| MAPE | ≤ 5.0% | Error persentase maksimum 5% |

Model juga harus **lebih baik dari model production sebelumnya** (RMSE lebih kecil dan R² lebih besar).

## File Konfigurasi

| File | Fungsi |
|------|--------|
| `ct_pipeline.py` | Script CT utama: train + compare + promote |
| `simulate_drift.py` | Simulasi data drift untuk testing |
| `alert_rules.yml` | Prometheus alerting rules |
| `.github/workflows/continuous_training.yml` | GitHub Actions workflow CT |

## Menjalankan CT Pipeline

### Manual via Terminal
```bash
python ct_pipeline.py manual
```

### Simulasi Data Drift
```bash
# Generate data dengan distribusi berbeda
python simulate_drift.py

# Push ke GitHub untuk trigger otomatis
git add data/processed/
git commit -m "data: add shifted data for CT simulation"
git push origin main
```

### Manual via GitHub Actions
Buka: `Actions → Continuous Training Pipeline - LK-12 → Run workflow`

## Hasil Simulasi LK-12

Simulasi dilakukan dengan data yang di-shift 24.9% ke atas (harga $249–$360 vs asli $199–$287):

| Metrik | Model Lama | Model Baru | Status |
|--------|-----------|-----------|--------|
| RMSE | — | 5.617 | ✅ Lolos |
| MAE | — | 4.3744 | ✅ Lolos |
| R² | — | 0.8506 | ✅ Lolos |
| MAPE | — | 1.3177% | ✅ Lolos |
| Dipromosikan | — | ✅ true | **Berhasil** |

Pipeline berjalan **tanpa intervensi manual** dalam 1 menit 50 detik.
