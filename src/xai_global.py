import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
m = mlflow.sklearn.load_model("models:/fraude@production")

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

_, Xte, _, _ = train_test_split(
    X, y, test_size=0.25, random_state=42
)

Xs = Xte.sample(400, random_state=0)

ex = shap.TreeExplainer(m)
sv = ex(Xs)[:, :, 1]

print("Valeur de base :", round(float(sv.base_values[0]), 4))

imp = pd.Series(
    np.abs(sv.values).mean(axis=0),
    index=Xs.columns
)

print(imp.sort_values(ascending=False).round(4).to_string())

imp.round(4).to_json("reports/shap_importance.json")

shap.plots.beeswarm(sv, show=False)
plt.tight_layout()
plt.savefig("reports/shap_global.png", dpi=120)
plt.close("all")

plt.scatter(
    Xs["montant"],
    sv[:, "montant"].values,
    c=Xs["nuit"],
    cmap="coolwarm",
    s=12
)

plt.xscale("log")
plt.xlabel("Montant (MAD)")
plt.ylabel("Contribution SHAP du montant")
plt.colorbar(label="nuit")
plt.tight_layout()
plt.savefig("reports/shap_montant.png", dpi=120)
plt.close("all")

print("SHAP global termine avec succes.")
