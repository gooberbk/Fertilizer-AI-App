# 🤖 Fertilizer AI : Prédiction de Rendement Agricole

Projet de Machine Learning M1 MIAGE / IA-IoT.
Ce projet utilise un modèle LightGBM pour prédire le rendement des cultures (tonnes/ha) en fonction de données environnementales et agricoles.


## 🎯 Problème
L'objectif est de créer un outil d'aide à la décision pour les agriculteurs. En fournissant des informations sur leur parcelle (type de sol, météo, engrais utilisé), l'application prédit le rendement final, permettant d'optimiser les ressources.

---

## Architecture du Projet

Ce n'est pas un simple script, mais une application web complète :

1.  **Le Cerveau (`agri_model_lgbm.joblib`)** : Un pipeline `scikit-learn` complet qui gère le pré-traitement (StandardScaler, OneHotEncoder) et la prédiction (modèle LightGBM).
2.  **Le Corps (`app.py`)** : Une API **Flask** qui "enveloppe" le modèle. Elle expose une route `/predict` qui reçoit des données JSON, les passe au modèle, et renvoie la prédiction.
3.  **Le Déploiement** : L'API est hébergée sur **Render** et connectée à ce dépôt GitHub, permettant des prédictions 24h/24.

---

## 🛠️ Stack Technique

* **Modèle ML** : LightGBM (LGBMRegressor)
* **Pré-traitement** : Scikit-learn (`Pipeline`, `ColumnTransformer`)
* **Serveur API** : Flask
* **Serveur de Production** : Gunicorn
* **Hébergement** : Render
* **Gestion de version** : Git & GitHub
