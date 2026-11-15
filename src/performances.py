from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

m = mlflow.sklearn.load_model("models:/fraude@production")

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

_, Xte, _, yte = train_test_split(
    X, y, test_size=0.25, random_state=42
)

pred = m.predict(Xte)
proba = m.predict_proba(Xte)[:, 1]

print("F1 :", round(f1_score(yte, pred), 4))
print("Precision :", round(precision_score(yte, pred), 4))
print("Rappel :", round(recall_score(yte, pred), 4))
print("AUC :", round(roc_auc_score(yte, proba), 4))
print("Matrice :", confusion_matrix(yte, pred).tolist())
