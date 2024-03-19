import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from joblib import dump

# IV.	Évaluation des Modèles
df_encoded = pd.read_csv('data_encoded.csv')

X = df_encoded.drop('Is Account Takeover', axis=1)
y = df_encoded['Is Account Takeover']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # <<<<< Modification 2:  aprés 4.2/ Premier test:
# # Modification pour utiliser l'échantillonnage stratifié
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Appliquer SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # <<<<< Modification 3:  aprés 4.3/ Second test:
# # Initialisation du modèle RandomForestClassifier
# # Définition de la grille d'hyperparamètres à tester
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'class_weight': [{0: 1, 1: 10}, 'balanced', None]  # Poids plus élevé sur la classe positive
# }
# rf = RandomForestClassifier(random_state=42)
# # Création de l'objet GridSearchCV
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
#                            scoring=make_scorer(recall_score), 
#                            cv=5, n_jobs=-1, verbose=2)
# # Ajustement du GridSearchCV sur les données d'entraînement
# grid_search.fit(X_train_smote, y_train_smote)
# # Affichage des meilleurs hyperparamètres
# print("Meilleurs hyperparamètres trouvés pour RandomForestClassifier: ", grid_search.best_params_)


# # Initialisation du modèle LogisticRegression
# # Configuration des hyperparamètres à tester
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],  # Valeurs de régularisation
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithmes d'optimisation
#     'class_weight': ['balanced']  # Utiliser 'balanced' pour corriger le déséquilibre des classes
# }

# rf = LogisticRegression(random_state=42)
# # Création de l'objet GridSearchCV
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
#                            scoring=make_scorer(recall_score), 
#                            cv=5, n_jobs=-1, verbose=2)
# # Ajustement du GridSearchCV sur les données d'entraînement
# grid_search.fit(X_train_smote, y_train_smote)
# # Affichage des meilleurs hyperparamètres
# print("Meilleurs hyperparamètres trouvés pour LogisticRegression: ", grid_search.best_params_)
# # # >>>>>>

# #  1. Vérification de la Distribution des Classes après SMOTE
# # Compter le nombre d'instances de chaque classe dans y_train_smote
# unique, counts = np.unique(y_train_smote, return_counts=True)
# class_distribution = dict(zip(unique, counts))

# print("Distribution des classes après SMOTE :")
# print(class_distribution)
# # >>>>>>

# ---------Premier test sur les algorithmes
# Exemple avec RandomForestClassifier
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Exactitude du modèle RandomForestClassifier: {accuracy:.2f}")

# # <<<<< Modification 2:  aprés 4.2/ Premier test:
# # ---------Second test sur les algorithmes
# # ////  Exemple avec RandomForestClassifier
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train_smote, y_train_smote)

# y_pred = model.predict(X_test)
# # <<<<< Modification 3:  aprés 4.3/ Second test:
# Hyperparamètres optimisés pour RandomForestClassifier
# optimal_params_rf = {
#     'class_weight': {0: 1, 1: 10},
#     'max_depth': 10,
#     'min_samples_leaf': 4,
#     'min_samples_split': 2,
#     'n_estimators': 300
# }

# <<<<< Modification 4:  aprés 4.4/ Troisième test:
# Hyperparamètres modifier tester
optimal_params_rf = {
    'class_weight': {0: 0.4, 1: 25},
    'max_depth': 3,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 200
}

# Création d'une nouvelle instance de RandomForestClassifier avec les hyperparamètres optimaux
model = RandomForestClassifier(**optimal_params_rf, random_state=42)

# Entraînement du modèle avec les données SMOTE
model.fit(X_train_smote, y_train_smote)

# Faire des prédictions avec le modèle optimisé
y_pred = model.predict(X_test)
# >>>>>>

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
# Afficher la matrice de confusion avec Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.ylabel('Vérité')
plt.xlabel('Prédiction')
plt.title('Matrice de Confusion')
plt.show()

# # Rapport de Classification
# print(classification_report(y_test, y_pred))

# # Définir la stratégie de validation croisée stratifiée
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # Spécifier l'AUC-ROC comme métrique de scoring
# scorer = make_scorer(roc_auc_score, needs_proba=True)
# # Effectuer la validation croisée stratifiée
# auc_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
# print("Scores AUC-ROC de la validation croisée : ", auc_scores)
# print("Moyenne des scores AUC-ROC : ", auc_scores.mean())

# Probabilité

# Obtenir les probabilités pour l'ensemble de test
probabilities = model.predict_proba(X_test)

# Prendre uniquement les probabilités d'appartenance à la classe frauduleuse (indice 1)
fraud_probabilities = probabilities[:, 1]

# Pour visualiser, par exemple, les 10 premières probabilités de fraude
print(fraud_probabilities[:10])

# Vous pouvez également ajouter ces probabilités comme une nouvelle colonne dans votre DataFrame
X_test_with_prob = X_test.copy()
X_test_with_prob['Fraud_Probability'] = fraud_probabilities

# # # Exemple avec LogisticRegression
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# <<<<< Modification 3:  aprés 4.3/ Second test:
# Hyperparamètres optimisés pour LogisticRegression
# optimal_params_rf = {
#     'C': 0.01, 
#     'class_weight': 'balanced', 
#     'solver': 'sag'
# }

optimal_params_rf = {
    'C': 8, 
    'class_weight': {0: 0.7, 1: 1}, 
    'solver': 'sag',
    'max_iter': 5000,
}

# Création d'une nouvelle instance de LogisticRegression avec les hyperparamètres optimaux
model = LogisticRegression(**optimal_params_rf)

# Entraînement du modèle avec les données SMOTE
model.fit(X_train_smote, y_train_smote)

# Faire des prédictions avec le modèle optimisé
y_pred = model.predict(X_test)
# # >>>>>>

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
# Afficher la matrice de confusion avec Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.ylabel('Vérité')
plt.xlabel('Prédiction')
plt.title('Matrice de Confusion')
plt.show()

# Rapport de Classification
print(classification_report(y_test, y_pred))

# # Définir la stratégie de validation croisée stratifiée
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # Spécifier l'AUC-ROC comme métrique de scoring
# scorer = make_scorer(roc_auc_score, needs_proba=True)
# # Effectuer la validation croisée stratifiée
# auc_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
# print("Scores AUC-ROC de la validation croisée : ", auc_scores)
# print("Moyenne des scores AUC-ROC : ", auc_scores.mean())

# Probabilité
# Obtenir les probabilités pour l'ensemble de test
probabilities = model.predict_proba(X_test)
# Prendre uniquement les probabilités d'appartenance à la classe frauduleuse (indice 1)
fraud_probabilities = probabilities[:, 1]
# Pour visualiser, par exemple, les 10 premières probabilités de fraude
print(fraud_probabilities[:10])
# Vous pouvez également ajouter ces probabilités comme une nouvelle colonne dans votre DataFrame
X_test_with_prob = X_test.copy()
X_test_with_prob['Fraud_Probability'] = fraud_probabilities
# Afficher le DataFrame avec la nouvelle colonne de probabilités
print(X_test_with_prob[['Fraud_Probability']].head())


# # # Exemple avec GaussianNB
# model = GaussianNB()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# # Matrice de confusion
# cm = confusion_matrix(y_test, y_pred)
# # Afficher la matrice de confusion avec Seaborn
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
# plt.ylabel('Vérité')
# plt.xlabel('Prédiction')
# plt.title('Matrice de Confusion')
# plt.show()

# # Rapport de Classification
# print(classification_report(y_test, y_pred))

# # Définir la stratégie de validation croisée stratifiée
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # Spécifier l'AUC-ROC comme métrique de scoring
# scorer = make_scorer(roc_auc_score, needs_proba=True)
# # Effectuer la validation croisée stratifiée
# auc_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
# print("Scores AUC-ROC de la validation croisée : ", auc_scores)
# print("Moyenne des scores AUC-ROC : ", auc_scores.mean())
# # >>>>>>

# 4.5/ Test avec des nouveaux hyperparamètres
# Sauvegarde du modèle random_forest
dump(model, 'random_forest_model.joblib')