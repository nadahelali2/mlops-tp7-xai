import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
m = mlflow.sklearn.load_model("models:/fraude@production")

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

_, Xte, _, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

r = permutation_importance(
    m, Xte, yte,
    scoring="f1",
    n_repeats=10,
    random_state=0
)

t = pd.DataFrame({
    "MDI (foret)": m.feature_importances_,
    "permutation": r.importances_mean,
    "ecart-type": r.importances_std
}, index=X.columns).round(3)

t = t.sort_values("permutation", ascending=False)

print(t.to_string())

t.to_json("reports/importance.json")

t["permutation"].plot.barh(
    xerr=t["ecart-type"],
    color="#4592EF"
)

plt.xlabel("Perte de F1 quand la variable est melangee")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("reports/permutation.png", dpi=120)

print("Graphique enregistre : reports/permutation.png")
