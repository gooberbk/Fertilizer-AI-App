import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 1. Initialiser l'application Flask
app = Flask(__name__)

# 2. Charger le "Cerveau" (notre pipeline .joblib)
#    Cela ne se produit qu'UNE SEULE FOIS au démarrage du serveur.
try:
    model = joblib.load('agri_model_lgbm.joblib')
    print("--- Modèle chargé avec succès ---")
except FileNotFoundError:
    print("ERREUR: Fichier 'agri_model_lgbm.joblib' introuvable.")
    print("Assurez-vous qu'il est dans le même dossier que app.py")
    exit()

# 3. Définir la "route" de prédiction
#    C'est l'URL que notre site web appellera (ex: /predict)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Fonction qui reçoit les données du formulaire web et renvoie une prédiction.
    """
    # Récupérer les données JSON envoyées par le formulaire
    data = request.json

    # --- C'est LA révolution de votre pipeline ---
    # 1. Convertir le JSON en DataFrame pandas
    #    Les clés du JSON DOIVENT correspondre aux noms de colonnes attendus.
    input_df = pd.DataFrame([data])

    # 2. Faire la prédiction
    #    Le pipeline gère TOUT : StandardScaler, OneHotEncoder, etc.
    prediction = model.predict(input_df)

    # 3. Renvoyer le résultat au format JSON
    return jsonify({'predicted_yield': prediction[0]})

# 4. Lancer le serveur
if __name__ == '__main__':
    # 'debug=True' permet au serveur de se recharger automatiquement
    # si vous modifiez le code.
    app.run(port=5000, debug=True)