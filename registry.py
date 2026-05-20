import mlflow
client = mlflow.MlflowClient(tracking_uri='http://127.0.0.1:5000')

run_id = '58c63f1f16e947d4bb85a6fff717de58'
model_uri = f'runs:/{run_id}/random_forest_model'

try:
    client.create_registered_model('AAPL-RF-Production')
except:
    pass

mv = client.create_model_version(name='AAPL-RF-Production', source=model_uri, run_id=run_id)
print(f'Version {mv.version} registered!')

client.set_registered_model_alias('AAPL-RF-Production', 'production', mv.version)
client.set_registered_model_alias('AAPL-RF-Production', 'staging', mv.version)
print('Alias production & staging diset!')
