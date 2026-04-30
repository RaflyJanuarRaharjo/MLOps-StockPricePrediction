import mlflow
import pandas as pd
import glob

mlflow.set_tracking_uri('sqlite:///mlflow.db')

print("="*55)
print("  VERIFIKASI INFERENSI — AAPL Stock Price Prediction")
print("  Rafly Januar Raharjo | 235150201111011")
print("="*55)

print("\nMemuat model Production...")
model = mlflow.pyfunc.load_model('models:/AAPL-RF-Production@production')
print("Model berhasil dimuat!")

files = sorted(glob.glob('data/processed/aapl_features_*.csv'))
df = pd.read_csv(files[-1], index_col='Date', parse_dates=True).dropna()
cols = ['Open','High','Low','Close','Volume','MA_7','MA_14','MA_30',
        'RSI_14','MACD','Signal','Hist','BB_upper','BB_lower',
        'Daily_Return','Close_lag1','Close_lag2','Close_lag5','Vol_MA_7']

X_sample = df[cols].tail(5)
predictions = model.predict(X_sample)

print("\nPrediksi 5 hari terakhir AAPL (Close T+1):")
print(f"{'Tanggal':<15} {'Prediksi':>12} {'Aktual':>12} {'Error':>10}")
print("-"*55)
for i, (date, pred) in enumerate(zip(X_sample.index, predictions)):
    actual = df['Target'].iloc[-(5-i)]
    error  = abs(pred - actual)
    print(f"{str(date.date()):<15} ${pred:>10.2f} ${actual:>10.2f} ${error:>8.2f}")

print("\nInferensi BERHASIL!")
print("="*55)