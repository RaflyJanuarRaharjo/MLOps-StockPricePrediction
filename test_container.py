import glob
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
files = sorted(glob.glob('/tmp/data/processed/aapl_features_*.csv'))
print('Files found:', files)
