import mlflow

client = mlflow.MlflowClient(tracking_uri='sqlite:///mlflow.db')
versions = client.search_model_versions("name='AAPL-RF-Production'")
print('Versi tersedia:')
for v in versions:
    print(f'  v{v.version} - Stage: {v.current_stage}')
client.set_registered_model_alias('AAPL-RF-Production', 'staging', '1')
client.set_registered_model_alias('AAPL-RF-Production', 'production', '1')
print('Alias production berhasil diset!')
print('Model AAPL-RF-Production v1 siap inferensi!')