# generate_load.py
# ============================================================
# Simulasi Beban Kerja (Load Generator) - LK-11
# Data diupdate sesuai harga AAPL terkini ~$280-$295
# Rafly Januar Raharjo - 235150201111011 | MLOps Kelas B
# ============================================================

import requests
import random
import time

API_URL        = "http://localhost:8000"
TOTAL_REQUESTS = 200
DELAY_BETWEEN  = 0.3

# Data AAPL terkini (Mei 2026, harga ~$280-$295)
SAMPLE_DATA = [
    {"open":287.0,"high":290.5,"low":285.0,"close":288.50,"volume":58000000,
     "ma_7":284.0,"ma_14":281.0,"ma_30":275.0,"rsi_14":65.0,
     "macd":4.2,"signal":3.8,"hist":0.4,"bb_upper":295.0,"bb_lower":272.0,
     "daily_return":0.008,"close_lag1":285.0,"close_lag2":283.0,"close_lag5":279.0,"vol_ma_7":55000000},

    {"open":280.0,"high":283.5,"low":278.0,"close":281.50,"volume":52000000,
     "ma_7":279.0,"ma_14":276.0,"ma_30":270.0,"rsi_14":58.0,
     "macd":3.5,"signal":3.2,"hist":0.3,"bb_upper":289.0,"bb_lower":265.0,
     "daily_return":0.004,"close_lag1":279.0,"close_lag2":277.0,"close_lag5":273.0,"vol_ma_7":50000000},

    {"open":293.0,"high":296.0,"low":291.0,"close":294.80,"volume":65000000,
     "ma_7":289.0,"ma_14":285.0,"ma_30":278.0,"rsi_14":72.0,
     "macd":5.8,"signal":5.0,"hist":0.8,"bb_upper":302.0,"bb_lower":278.0,
     "daily_return":0.012,"close_lag1":291.0,"close_lag2":289.0,"close_lag5":284.0,"vol_ma_7":61000000},

    {"open":275.0,"high":278.0,"low":273.0,"close":276.20,"volume":48000000,
     "ma_7":277.0,"ma_14":275.0,"ma_30":270.0,"rsi_14":48.0,
     "macd":2.1,"signal":2.5,"hist":-0.4,"bb_upper":286.0,"bb_lower":264.0,
     "daily_return":-0.005,"close_lag1":278.0,"close_lag2":280.0,"close_lag5":282.0,"vol_ma_7":47000000},

    {"open":284.0,"high":287.0,"low":282.0,"close":285.90,"volume":54000000,
     "ma_7":282.0,"ma_14":279.0,"ma_30":273.0,"rsi_14":62.0,
     "macd":4.5,"signal":4.0,"hist":0.5,"bb_upper":293.0,"bb_lower":269.0,
     "daily_return":0.006,"close_lag1":283.0,"close_lag2":281.0,"close_lag5":277.0,"vol_ma_7":52000000},
]


def send_predict(payload):
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        return resp.status_code, resp.json()
    except Exception as e:
        return 0, {"error": str(e)}


def send_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code
    except:
        return 0


def send_predict_latest():
    try:
        resp = requests.get(f"{API_URL}/predict-latest", timeout=5)
        return resp.status_code, resp.json()
    except Exception as e:
        return 0, {"error": str(e)}


def main():
    print(f"=" * 60)
    print(f"  AAPL Load Generator - LK-11 Monitoring")
    print(f"  Target: {API_URL}")
    print(f"  Total requests: {TOTAL_REQUESTS}")
    print(f"=" * 60)

    success = 0
    fail    = 0

    for i in range(1, TOTAL_REQUESTS + 1):

        if i % 15 == 0:
            code = send_health()
            print(f"[{i:3d}/{TOTAL_REQUESTS}] GET  /health        → {code}")

        if i % 30 == 0:
            code, body = send_predict_latest()
            pred = body.get("prediction", "N/A")
            print(f"[{i:3d}/{TOTAL_REQUESTS}] GET  /predict-latest → {code} | pred=${pred}")

        sample = random.choice(SAMPLE_DATA).copy()
        for key in ["open", "high", "low", "close", "close_lag1", "close_lag2", "close_lag5"]:
            sample[key] += random.uniform(-4.0, 4.0)
        sample["rsi_14"]       = max(10, min(90, sample["rsi_14"] + random.uniform(-5, 5)))
        sample["daily_return"] += random.uniform(-0.005, 0.005)

        code, body = send_predict(sample)
        if code == 200:
            success += 1
            pred = body.get("prediction", "?")
            print(f"[{i:3d}/{TOTAL_REQUESTS}] POST /predict        → {code} | AAPL=${pred:.2f}")
        else:
            fail += 1
            print(f"[{i:3d}/{TOTAL_REQUESTS}] POST /predict        → ERROR {code}: {body}")

        time.sleep(DELAY_BETWEEN)

    print(f"=" * 60)
    print(f"  Selesai! Sukses: {success} | Gagal: {fail}")
    print(f"  Buka Grafana: http://localhost:3000 (admin/admin123)")
    print(f"=" * 60)


if __name__ == "__main__":
    main()