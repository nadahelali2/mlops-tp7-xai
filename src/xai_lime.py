import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
m = mlflow.sklearn.load_model("models:/fraude@production")

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

Xtr, _, _, _ = train_test_split(
    X, y, test_size=0.25, random_state=42
)

CAS = pd.DataFrame([{
    "montant": 1800.0,
    "anciennete": 30,
    "freq_24h": 4,
    "nuit": 1
}])

def expliquer(graine, n):
    ex = LimeTabularExplainer(
        Xtr.values,
        feature_names=list(X.columns),
        class_names=["normal", "fraude"],
        mode="classification",
        random_state=graine
    )
    return ex.explain_instance(
        CAS.values[0],
        m.predict_proba,
        num_features=4,
        num_samples=n
    )

e = expliquer(0, 2000)

print("Regles :", [(r, round(w, 3)) for r, w in e.as_list()])

e.save_to_file("reports/lime_benali.html")

fig = e.as_pyplot_figure()
fig.tight_layout()
fig.savefig("reports/lime_benali.png", dpi=120)
plt.close(fig)

for n in [500, 2000]:
    print(f"--- {n} voisins ---")
    print("graine | montant | nuit | anciennete | freq_24h")

    for graine in range(5):
        d = dict(expliquer(graine, n).as_list())
        w = [
            next(v for k, v in d.items() if col in k)
            for col in X.columns
        ]
        print(
            f"{graine} | {w[0]:+.3f} | {w[3]:+.3f} | "
            f"{w[1]:+.3f} | {w[2]:+.3f}"
        )

print("LIME termine avec succes.")
