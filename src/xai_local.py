import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
m = mlflow.sklearn.load_model("models:/fraude@production")

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

_, Xte, _, _ = train_test_split(
    X, y, test_size=0.25, random_state=42
)

CAS = pd.DataFrame([{
    "montant": 1800.0,
    "anciennete": 30,
    "freq_24h": 4,
    "nuit": 1
}])

ex = shap.TreeExplainer(m)
proba = m.predict_proba(Xte)[:, 1]
idx = Xte.index[proba.argmax()]

for nom, x in [
    ("benali", CAS),
    (f"test_{idx}", Xte.loc[[idx]])
]:
    e = ex(x)[0, :, 1]

    print(f"--- {nom} :", x.iloc[0].to_dict())
    print("Valeur de base :", round(float(e.base_values), 4))

    for col, c in sorted(
        zip(x.columns, e.values),
        key=lambda t: -abs(t[1])
    ):
        print(f" {col:11s} {c:+.4f}")

    print(
        "Base + somme :",
        round(float(e.base_values + e.values.sum()), 4),
        "| Prediction :",
        round(m.predict_proba(x)[0, 1], 4)
    )

    shap.plots.waterfall(e, show=False)
    plt.tight_layout()
    plt.savefig(f"reports/shap_{nom}.png", dpi=120)
    plt.close("all")

print("SHAP local termine avec succes.")
