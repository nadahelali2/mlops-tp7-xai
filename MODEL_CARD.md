# Model Card - fraude (version 1)

## Modele
Nom et version : fraude v1, alias production.
Famille : RandomForestClassifier, 500 arbres, profondeur maximale 16.
Run d'origine : 311adcac08ee43b6b400ba940e8e8330.

## Objectif et usage prevu
Detecter les transactions bancaires potentiellement frauduleuses afin de faciliter leur verification.

## Usages exclus
Ne pas bloquer definitivement un compte ou sanctionner un client sans verification humaine.

## Donnees
Fichier : data/transactions.csv.
Transactions : 5000.
Fraudes : 568 (11.36%).
Empreinte MD5 : cf9f6f86c141e7df32c61715c493483f.
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
Run des explications : ef57992bef9244e69c4dd02704f2be58.
Run du modele : 311adcac08ee43b6b400ba940e8e8330.
Auteure : Nada Helali.
Date : 2026-11-15.
Prochaine revue : avant tout deploiement reel.
