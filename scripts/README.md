# Projet de ML pour la détection de connexions frauduleuses

## Aperçu
Ce projet vise à construire et évaluer un modèle de détection de fraude en utilisant des techniques d'apprentissage automatique. Le projet est structuré en plusieurs étapes : création du dataset, exploration des données, prétraitement, encodage, entraînement et évaluation du modèle, et réalisation de prédictions avec un modèle formé utilisable à travers une API.

## Structure du Projet
- `creation_dataset.py` : Script pour créer un dataset plus petit et équilibré à partir du dataset original pour une exploration et un entraînement du modèle efficaces.
- `data_exploration.py` : Script pour l'exploration initiale des données afin de comprendre les caractéristiques du dataset, y compris la distribution des classes, les valeurs manquantes, etc.
- `data_preprocessing.py` : Script pour le prétraitement du dataset, y compris la gestion des valeurs manquantes, feature engineering, etc.
- `data_encoding.py` : Script pour encoder les features dans un format adapté aux modèles d'apprentissage automatique.
- `model_test_training.py` : Script pour former, évaluer et affiner les modèles d'apprentissage automatique.
- `data_preprocessing_prod.py` : Script pour le prétraitement des nouvelles données en production envoyé via l'API avant de faire des prédictions.
- `api.py` : Script Flask API qui reçoit des données, les prétraite et utilise le modèle formé pour faire des prédictions.

## Exécution des Scripts
1. **Création du Dataset** : Exécutez `creation_dataset.py` pour générer un dataset équilibré et plus petit à partir du dataset original. Cette étape est cruciale pour une exploration et un entraînement efficaces.
`python creation_dataset.py`

2. **Exploration des Donnée**: Exécutez data_exploration.py pour effectuer une analyse exploratoire initiale du dataset.
`python data_exploration.py`

3. **Prétraitement des Données**: Traitez le dataset en exécutant data_preprocessing.py, qui prépare les données et fait appel à data_encoding.py pour les encoder.

4. **Entraînement et Évaluation du Modèle**: Formez et évaluez les modèles en utilisant model_test_training.py. Ce script inclut également à la fin la persistance du modèle formé pour le déployer
`python model_test_training.py`

## API pour les Prédictions
Après avoir formé le modèle et préparé les scripts de l'API, vous pouvez exécuter l'API Flask pour faire des prédictions en temps réel sur les nouvelles données de connexion.

### Lancer l'API
1. Assurez-vous que Flask est installé dans votre environnement.
2. Exécutez le script `api.py` pour démarrer l'API.
   `python api.py`
Cela lancera un serveur Flask local accessible à l'adresse par défaut http://127.0.0.1:5000.

**Utiliser l'API**
Envoyez une requête POST à l'API avec les données de connexion au format JSON à la route /evaluate-connection. Par exemple :

```python
curl -X POST http://127.0.0.1:5000/evaluate-connection \
-H "Content-Type: application/json" \
-d '{
      "Login Timestamp": "2021-06-21T15:32:00",
      "User Agent String": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }'
```
L'API traitera les données reçues, utilisera le modèle formé pour prédire la probabilité de fraude, et retournera cette probabilité dans la réponse.

**Fichier api.py**
Le fichier api.py contient la définition de l'API Flask, incluant la route /evaluate-connection qui accepte les données de connexion, les prétraite à l'aide de data_preprocessing_prod.py, et renvoie la probabilité de fraude calculée par le modèle.