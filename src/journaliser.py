import hashlib
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")

NOM = "fraude"
ALIAS = "production"

client = MlflowClient()
VERSION = client.get_model_version_by_alias(NOM, ALIAS)
m = mlflow.sklearn.load_model(f"models:/{NOM}@{ALIAS}")

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

_, Xte, _, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

def empreinte(chemin):
    return hashlib.md5(Path(chemin).read_bytes()).hexdigest()

pred = m.predict(Xte)

mlflow.set_experiment("tp7-explicabilite")

with mlflow.start_run(
    run_name=f"xai-{NOM}-v{VERSION.version}"
) as run:

    mlflow.set_tags({
        "modele": NOM,
        "alias": ALIAS,
        "modele_version": VERSION.version,
        "modele_run_id": VERSION.run_id,
        "donnees": "data/transactions.csv",
        "donnees_md5": empreinte("data/transactions.csv"),
        "methodes": "permutation ; SHAP TreeExplainer ; LIME"
    })

    mlflow.log_params({
        "shap_echantillon": 400,
        "lime_num_samples": 2000,
        "lime_random_state": 0,
        "permutation_n_repeats": 10
    })

    mlflow.log_metrics({
        "f1": f1_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred)
    })

    for f in sorted(Path("reports").glob("*.*")):
        if f.suffix not in (".png", ".html", ".json"):
            continue
        mlflow.log_artifact(
            str(f),
            artifact_path="explications"
        )

    Path("reports/run_xai.txt").write_text(
        run.info.run_id
    )

client.set_model_version_tag(
    NOM,
    VERSION.version,
    "explications_run",
    run.info.run_id
)

print("Run :", run.info.run_id)
print("Modele :", NOM)
print("Version :", VERSION.version)
print("Explications archivees avec succes.")
