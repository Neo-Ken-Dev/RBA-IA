from flask import Flask, request, jsonify
import data_preprocessing_prod as pp
import data_encoding as de
from joblib import load

app = Flask(__name__)

# Charger les modèles sauvegardés
model = load('random_forest_model.joblib')

@app.route('/evaluate-connection', methods=['POST'])
def process_request():
    data = request.json

    # Appliquer le prétraitement
    df_processed = pp.process_data(data)

    # Appliquer l'encodage
    df_encoded = de.encode_data(df_processed)

    # Utiliser le modèle pour faire des prédictions
    # Exemple d'utilisation du modèle RandomForest
    predictions = model.predict_proba(df_encoded)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)