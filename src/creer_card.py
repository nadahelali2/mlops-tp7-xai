from pathlib import Path
from datetime import date
import hashlib
import pandas as pd
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

v = client.get_model_version_by_alias("fraude", "production")
run = client.get_run(v.run_id)
run_xai = Path("reports/run_xai.txt").read_text().strip()

df = pd.read_csv("data/transactions.csv")
md5 = hashlib.md5(Path("data/transactions.csv").read_bytes()).hexdigest()

card = f"""# Model Card - fraude (version {v.version})

## Modele
Nom et version : fraude v{v.version}, alias production.
Famille : RandomForestClassifier, 500 arbres, profondeur maximale 16.
Run d'origine : {v.run_id}.

## Objectif et usage prevu
Detecter les transactions bancaires potentiellement frauduleuses afin de faciliter leur verification.

## Usages exclus
Ne pas bloquer definitivement un compte ou sanctionner un client sans verification humaine.

## Donnees
Fichier : data/transactions.csv.
Transactions : {len(df)}.
Fraudes : {int(df['fraude'].sum())} ({df['fraude'].mean()*100:.2f}%).
Empreinte MD5 : {md5}.
Origine : donnees synthetiques du TP.

## Performances
F1 : 0.8669.
Precision : 0.8837.
Rappel : 0.8507.
AUC : 0.9872.
Jeu de test : 1250 transactions.

## Methodes d'explicabilite
Globale : importance par permutation et SHAP.
Locale : SHAP TreeExplainer et LIME.
Variables principales : montant, anciennete, nuit, freq_24h.
Mme Benali : probabilite de fraude de 92.4%, expliquee principalement par la nuit et le montant.

## Comparaison avec un modele interpretable
Foret aleatoire : F1 = 0.8669.
Regression logistique avec log10(montant) : F1 = 0.8593.
La regression logistique est plus directement interpretable, avec une performance proche.

## Limites et risques connus
Donnees synthetiques, generalisation non verifiee sur des transactions reelles.
LIME peut varier selon le tirage aleatoire.
Les valeurs SHAP ne prouvent pas une causalite.

## Recommandations d'utilisation et de surveillance
Verifier humainement les alertes importantes.
Surveiller le rappel, les faux positifs et la derivee des donnees.
Reevaluer le modele avant toute utilisation reelle.

## Tracabilite
Run des explications : {run_xai}.
Run du modele : {v.run_id}.
Auteure : Nada Helali.
Date : {date.today().isoformat()}.
Prochaine revue : avant tout deploiement reel.
"""

Path("MODEL_CARD.md").write_text(card, encoding="utf-8")
print("MODEL_CARD.md cree avec succes.")
print("Version :", v.version)
print("Run XAI :", run_xai)
