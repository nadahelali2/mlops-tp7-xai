import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

mlflow.set_tracking_uri("http://127.0.0.1:5000")
m = mlflow.sklearn.load_model("models:/fraude@production")

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

CAS = pd.DataFrame([{
    "montant": 1800.0,
    "anciennete": 30,
    "freq_24h": 4,
    "nuit": 1
}])

res = {}

def note(nom, modele, Xt):
    pred = modele.predict(Xt)
    res[nom] = {
        "f1": round(f1_score(yte, pred), 4),
        "fraudes_detectees": int(((pred == 1) & (yte == 1)).sum())
    }
    print(
        f'{nom:35s} F1 = {res[nom]["f1"]:.4f} | '
        f'fraudes detectees : {res[nom]["fraudes_detectees"]} / {int(yte.sum())}'
    )

def logm(d):
    d = d.copy()
    d["log_montant"] = np.log10(d.pop("montant"))
    return d

note("foret fraude@production", m, Xte)

lr_brut = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
).fit(Xtr, ytr)

note("logistique, montant brut", lr_brut, Xte)

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
).fit(logm(Xtr), ytr)

note("logistique, log10(montant)", lr, logm(Xte))

coef = {
    c: round(float(v), 2)
    for c, v in zip(logm(Xtr).columns, lr[-1].coef_[0])
}

print("Coefficients standardises :", coef)

cote = np.exp(lr[-1].coef_[0] / lr[0].scale_)

print(
    "Cote anciennete :", round(cote[0], 3),
    "| Cote freq_24h :", round(cote[1], 2)
)

for p in [2, 3, 6, None]:
    t = DecisionTreeClassifier(
        max_depth=p, random_state=0
    ).fit(Xtr, ytr)

    note(
        f"arbre profondeur {p} ({t.get_n_leaves()} feuilles)",
        t, Xte
    )

t3 = DecisionTreeClassifier(
    max_depth=3, random_state=0
).fit(Xtr, ytr)

print("\nRegles de l'arbre profondeur 3 :")
print(export_text(t3, feature_names=list(X.columns), decimals=1))

print(
    "Mme Benali : feuille",
    t3.apply(CAS)[0],
    "| P(fraude) =",
    round(t3.predict_proba(CAS)[0, 1], 3)
)

with open("reports/interpretables.json", "w") as f:
    json.dump(res, f, indent=2)

print("Activite 5 terminee avec succes.")
