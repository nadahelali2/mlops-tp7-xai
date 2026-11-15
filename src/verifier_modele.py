import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()
v = client.get_model_version_by_alias("fraude", "production")
m = mlflow.sklearn.load_model("models:/fraude@production")

print("Modele : fraude")
print("Version :", v.version)
print("Alias : production")
print("Type :", type(m).__name__)
print("Nombre d'arbres :", m.n_estimators)
print("Profondeur maximale :", m.max_depth)
print("Run :", v.run_id)
