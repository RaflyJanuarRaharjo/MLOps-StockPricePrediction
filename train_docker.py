import glob, warnings, numpy as np, pandas as pd, mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('AAPL-RandomForest')

FEATURE_COLS = ['Open','High','Low','Close','Volume','MA_7','MA_14','MA_30',
    'RSI_14','MACD','Signal','Hist','BB_upper','BB_lower',
    'Daily_Return','Close_lag1','Close_lag2','Close_lag5','Vol_MA_7']

files = sorted(glob.glob('/tmp/data/processed/aapl_features_*.csv'))
df = pd.read_csv(files[-1], index_col='Date', parse_dates=True).dropna()
X = df[FEATURE_COLS].values
y = df['Target'].values

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

params = {'n_estimators':350,'max_depth':10,'min_samples_split':10,'min_samples_leaf':4,'max_features':0.7}
model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

with mlflow.start_run(run_name='RF-Shallow-Trees-Docker') as run:
    for k,v in params.items():
        mlflow.log_param(k, v)
    mlflow.log_metric('rmse', round(rmse,4))
    mlflow.log_metric('mae', round(mae,4))
    mlflow.log_metric('r2_score', round(r2,4))
    sig = mlflow.models.infer_signature(X, model.predict(X))
    mlflow.sklearn.log_model(model, 'random_forest_model', signature=sig)
    run_id = run.info.run_id
    print('Run ID:', run_id)
