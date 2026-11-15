import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp7-preparation")

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

with mlflow.start_run(run_name="rf-500-16"):
    m = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        random_state=42
    )
    m.fit(Xtr, ytr)

    f1 = f1_score(yte, m.predict(Xte))

    mlflow.log_params({
        "n_estimators": 500,
        "max_depth": 16
    })
    mlflow.log_metric("f1", f1)

    info = mlflow.sklearn.log_model(
        m,
        name="model",
        input_example=Xte.head(3),
        registered_model_name="fraude",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

MlflowClient().set_registered_model_alias(
    "fraude", "production", info.registered_model_version
)

print("Modele : fraude")
print("Version :", info.registered_model_version)
print("Alias : production")
print("F1 :", round(f1, 4))
