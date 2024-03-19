import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from user_agents import parse
import category_encoders as ce
from sklearn.metrics import confusion_matrix


# ---- 1) Créer un workspace--------------

# ---- 2) Création d’une copie de la data--------------
# Lecture du dataset
# df = pd.read_csv('C:/Users/lemon/OneDrive/Bureau/Ken/3wa/Diplome/Dossier_data/RBA_project/data/dataset_kaggle/rba-dataset.csv')

# <<<<< Modification 1 avant exploration:
# Réduction du Dataset à 1 000 000 de lignes 
# df_to_save = df[0:1000000]
# df_to_save.to_csv('smallDataset.csv')
# df = pd.read_csv('C:/Users/lemon/OneDrive/Bureau/Ken/3wa/Diplome/Dossier_data/RBA_project/scripts/smallDataset.csv')
# >>>>>>

# isoler un dataset d’entrainement et de test
# train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# copie de mon set d'entrainement avant l'exploration
# df_exploration = train_set.copy()

# ---- 3) Vérification de la data--------------
# 1 Première visualisation :
# print(df_exploration.head(10))

# 2 Nom et type des colonnes :
# description rapide de la data
# print(df_exploration.info())

# 3/ Compréhension des entrées null dans le dataset.











# ----Reduction et lecture de mon dataset-----
# Dataset Complet
# df = pd.read_csv('C:/Users/lemon/OneDrive/Bureau/Ken/3wa/Diplome/Dossier_data/RBA_project/data/dataset_kaggle/rba-dataset.csv')



# # cration dataset avec take over et 1000000 non take over
# # Séparer les instances où Is Account Takeover est True
# takeover_true = df[df['Is Account Takeover'] == True]

# # Séparer les instances où Is Account Takeover est False
# takeover_false = df[df['Is Account Takeover'] == False]

# # Échantillonnage aléatoire de 1 000 000 d'instances False
# takeover_false_sample = takeover_false.sample(n=1000000, random_state=42)

# # Concaténer les instances True et l'échantillon aléatoire des instances False
# new_dataset = pd.concat([takeover_true, takeover_false_sample])

# # Mélanger les lignes du nouveau dataset pour éviter toute séquence
# new_dataset = new_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# new_dataset.to_csv('smallDataset.csv')

# fin création dataset avec take over et 1000000 non take over
df = pd.read_csv('C:/Users/lemon/OneDrive/Bureau/Ken/3wa/Diplome/Dossier_data/RBA_project/scripts/smallDataset.csv')


# Dataset 1 000 000 
# df_to_save = df[0:1000000]
# df_to_save.to_csv('smallDataset.csv')
# df = pd.read_csv('C:/Users/lemon/OneDrive/Bureau/Ken/3wa/Diplome/Dossier_data/RBA_project/scripts/smallDataset.csv')

# Dataset avec uniquement les connections frauduleuses
# df_only_take_over = df[df['Is Account Takeover'] == True]
# df_only_take_over.to_csv('TakeoverDataset.csv')
# df_fraud = pd.read_csv('C:/Users/lemon/OneDrive/Bureau/Ken/3wa/Diplome/Dossier_data/RBA_project/scripts/TakeoverDataset.csv')

# ----Split et copie du dataset-----
# Séparation du train et test set
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# Copie du train_set
df_exploration = train_set.copy()

# ----Exploration du dataset----- 
# print(df_exploration.info())
# print(df_exploration.isnull().sum())








# print(df_exploration['OS Name and Version'].value_counts())
# Display count of RTT values by device type
# dfBis_count = df_exploration.groupby('Device Type')['Round-Trip Time [ms]'].count()
# print(dfBis_count)


# ///////---analyse des valeurs null---\\\\\\\
# <<<<<----analyse des regions null
# null_regions = df_exploration[df_exploration['Region'].isnull()]

# # Distribution des regions null par pays:
# print(null_regions['Country'].value_counts())

# # Distribution des regions null par type d'appareil:
# print(null_regions['Device Type'].value_counts())

# # Distribution des regions null parville:
# print(null_regions['City'].value_counts())

# # ****Compare the % of successful connections between datas with and without Region****
# null_regions = df_exploration[df_exploration['Region'].isnull()]
# total_logins_no_region = len(null_regions)
# login_success_count_no_region = null_regions['Login Successful'].value_counts()
# rate_success_no_region = (login_success_count_no_region[True] / total_logins_no_region) * 100
# rate_failure_no_region = (login_success_count_no_region[False] / total_logins_no_region) * 100

# not_null_regions = df_exploration[df_exploration['Region'].notnull()]
# total_logins_with_region = len(not_null_regions)
# login_success_count_with_region = not_null_regions['Login Successful'].value_counts()
# rate_success_with_region = (login_success_count_with_region[True] / total_logins_with_region) * 100
# rate_failure_with_region = (login_success_count_with_region[False] / total_logins_with_region) * 100

# data = {
#     "Condition": ["Sans Region", "Avec Region"],
#     "Total Logins": [total_logins_no_region, total_logins_with_region],
#     "Login Successful": [login_success_count_no_region[True], login_success_count_with_region[True]],
#     "Login Failure": [login_success_count_no_region[False], login_success_count_with_region[False]],
#     "Success Rate (%)": [rate_success_no_region, rate_success_with_region],
#     "Failure Rate (%)": [rate_failure_no_region, rate_failure_with_region]
# }

# # Convert data to DataFrame
# comparison_df_login_success = pd.DataFrame(data)
# print(comparison_df_login_success)

# # ****Compare the % of fraud connections between datas with and without Region****

# null_regions = df_exploration[df_exploration['Region'].isnull()]
# total_logins_no_region = len(null_regions)
# takeover_count_no_region = null_regions['Is Account Takeover'].value_counts()
# rate_fraud_no_region = (takeover_count_no_region[True] / total_logins_no_region) * 100
# rate_not_fraud_no_region = (takeover_count_no_region[False] / total_logins_no_region) * 100

# not_null_regions = df_exploration[df_exploration['Region'].notnull()]
# total_logins_with_region = len(not_null_regions)
# takeover_count_with_region = not_null_regions['Is Account Takeover'].value_counts()
# rate_fraud_with_region = (takeover_count_with_region[True] / total_logins_with_region) * 100
# rate_not_fraud_with_region = (takeover_count_with_region[False] / total_logins_with_region) * 100

# data = {
#     "Condition": ["Sans Region", "Avec Region"],
#     "Total Logins": [total_logins_no_region, total_logins_with_region],
#     "Login Fraud": [takeover_count_no_region[True], takeover_count_with_region[True]],
#     "Login Not Fraud": [takeover_count_no_region[False], takeover_count_with_region[False]],
#     "Fraud Rate (%)": [rate_fraud_no_region, rate_fraud_with_region],
#     "Not Fraud Rate (%)": [rate_not_fraud_no_region, rate_not_fraud_with_region]
# }

# # Convert data to DataFrame
# comparison_df_login_fraud = pd.DataFrame(data)
# print(comparison_df_login_fraud)


# # Séparer les lignes avec et sans région
# with_region = df.dropna(subset=['Region'])
# without_region = df[df['Region'].isnull()]

# # Calculer le pourcentage de 'Is Account Takeover' à True pour les lignes avec région
# perc_with_region_true = (with_region['Is Account Takeover'].sum() / len(with_region)) * 100

# # Calculer le pourcentage de 'Is Account Takeover' à True pour les lignes sans région
# perc_without_region_true = (without_region['Is Account Takeover'].sum() / len(without_region)) * 100

# print(f"Pourcentage avec région et 'Is Account Takeover' à True: {perc_with_region_true}%")
# print(f"Pourcentage sans région et 'Is Account Takeover' à True: {perc_without_region_true}%")

# # Ratio des pourcentages
# if perc_with_region_true > 0:  # Pour éviter la division par zéro
#     ratio = perc_without_region_true / perc_with_region_true
#     print(f"Ratio des pourcentages (sans région / avec région) : {ratio}")
# else:
#     print("Le ratio ne peut pas être calculé car le dénominateur est zéro.")

# # Visualisation (nécessite matplotlib)
# plt.bar(['Avec région', 'Sans région'], [perc_with_region_true, perc_without_region_true])
# plt.ylabel('Pourcentage de Is Account Takeover à True')
# plt.title('Comparaison des pourcentages de Fraude avec et sans region')
# plt.show()

# # Pourcentage de sans région sur tout le dataset:
# null_regions = df_exploration[df_exploration['Region'].isnull()]
# total_logins_no_region = len(null_regions)
# total_entries = len(df_exploration)
# print("total_entries:", total_entries)
# print("total_logins_no_region:", total_logins_no_region)

# rate_no_region = (total_logins_no_region / total_entries) * 100
# # print("rate of data no region:", rate_no_region)
# ---->>>>>>>>>>

# <<<<<----analyse des villes null
# To get a quick look at the first few rows with null cities
# null_cities = df_exploration[df_exploration['City'].isnull()]

# # Analyze the distribution of countries for null cities
# print("Analyze the distribution of countries for null cities")
# print(null_cities['Country'].value_counts())

# # Check for device types
# print("Check for device types")
# print(null_cities['Device Type'].value_counts())
# # Check if specific cities or IP addresses are more associated with null cities
# print("check if specific cities or IP addresses are more associated with null cities")

# print(null_cities['Region'].value_counts())
# print(null_cities['IP Address'].value_counts())
# # Example: Check if null cities occur more frequently with certain login success rates
# print("Login Successfuls for data without city")
# print(null_cities['Login Successful'].value_counts())

# # ****Comparer le % des connection succes entre les entrés avec et sans ville****
# null_cities = df_exploration[df_exploration['City'].isnull()]
# total_logins_without_city = len(null_cities)
# login_success_count_without_cities = null_cities['Login Successful'].value_counts()
# percentage_success_without_cities = (login_success_count_without_cities[True] / total_logins_without_city) * 100
# percentage_failure_without_cities = (login_success_count_without_cities[False] / total_logins_without_city) * 100

# not_null_cities = df_exploration[df_exploration['City'].notnull()]
# total_logins_with_cities = len(not_null_cities)
# login_success_count_with_cities = not_null_cities['Login Successful'].value_counts()
# percentage_success_with_cities = (login_success_count_with_cities[True] / total_logins_with_cities) * 100
# percentage_failure_with_cities = (login_success_count_with_cities[False] / total_logins_with_cities) * 100

# data = {
#     "Condition": ["Sans Ville", "Avec Ville"],
#     "Total Logins": [total_logins_without_city, total_logins_with_cities],
#     "Login Successful": [login_success_count_without_cities[True], login_success_count_with_cities[True]],
#     "Login Failure": [login_success_count_without_cities[False], login_success_count_with_cities[False]],
#     "Success Rate (%)": [percentage_success_without_cities, percentage_success_with_cities],
#     "Failure Rate (%)": [percentage_failure_without_cities, percentage_failure_with_cities]
# }

# # Convert data to DataFrame
# comparison_df_login_success = pd.DataFrame(data)
# print("Comparison dataframe for login success for data without city")
# print(comparison_df_login_success)

# ****Comparer le % des connection frauduleuse entre les entrés avec et sans cities****
# null_cities =df_exploration[df_exploration['City'].isnull()]
# print("Is Account Takeover for data without city")
# print(null_cities['Is Account Takeover'].value_counts())

# # Pourcentage de sans cities sur tout le dataset:
# null_cities = df_exploration[df_exploration['City'].isnull()]
# total_logins_without_cities = len(null_cities)
# total_entries = len(df_exploration)
# print("total_entries:", total_entries)
# print("total_logins_without_cities:", total_logins_without_cities)
# percentage_without_city = (total_logins_without_cities / total_entries) * 100
# print("Percentage of data without ville:", percentage_without_city)


# ///////---Que faire de la datat colonne par colonne---\\\\\\\
# # Colonne par colonne modifier la data?
# # --------index-----------
df_exploration = df_exploration.drop(columns=['index'])

# # # --------User ID-----------
# # Afficher les données pour l'utilisateur avec l'ID spécifique avant transformation
# user_ip_region = df.loc[df['User ID'] == -4324475583306591935, ['IP Address', 'Region']]

# print(user_ip_region)

# # Le user ID en l'état n'est pas trés utile en revanche nous pouvons en tirer des informations intéressante comme le nombre de connexion et le taux de sonnection succes 
# # Calculer le nombre total de connexions par utilisateur
# total_connexions = df_exploration.groupby('User ID').size().reset_index(name='Total Connections')
# # Calculer le nombre de connexions réussies par utilisateur
# succes_connexions = df_exploration[df_exploration['Login Successful'] == True].groupby('User ID').size().reset_index(name='Successful connections')

# # Fusionner les deux DataFrames sur 'User ID'
# user_stats = pd.merge(total_connexions, succes_connexions, on='User ID', how='left')

# # Remplacer les NaN par 0 pour les utilisateurs sans connexions réussies
# user_stats['Successful connections'] = user_stats['Successful connections'].fillna(0)

# # Calculer le taux de succès de connexion
# user_stats['Successful rate connections'] = (user_stats['Successful connections'] / user_stats['Total Connections']) * 100

# df_exploration = pd.merge(df_exploration, user_stats[['User ID', 'Total Connections', 'Successful rate connections']], on='User ID', how='left')

# # Afficher le résultat
# # Sélectionner une ligne par 'User ID'
# df_unique_users = df_exploration.drop_duplicates(subset=['User ID'])

# # Trier par 'Total Connections' pour obtenir les 10 premiers utilisateurs distincts
# top_10_unique_users = df_unique_users.sort_values(by='Total Connections', ascending=False).head(10)

# # Afficher les résultats
# print(top_10_unique_users[['User ID', 'Total Connections', 'Successful rate connections']])
# ********finalement on drop**********
df_exploration = df_exploration.drop(columns=['User ID'])

# --------colonne time stamp--------
# Afficher les premières
# print(df['Login Timestamp'])

# # Convertir la colonne 'Login Timestamp' en datetime
df_exploration['Login Timestamp'] = pd.to_datetime(df_exploration['Login Timestamp'])

# Extraire le jour et l'heure
df_exploration['Day'] = df_exploration['Login Timestamp'].dt.day
df_exploration['Hour'] = df_exploration['Login Timestamp'].dt.hour

# Déterminer si c'est le weekend
df_exploration['Is Weekend'] = df_exploration['Login Timestamp'].dt.dayofweek >= 5

# Déterminer si l'heure est en dehors des heures de bureau
df_exploration['Outside Office Hours'] = (df_exploration['Hour'] < 7) | (df_exploration['Hour'] > 21)

df_exploration = df_exploration.drop(columns=['Login Timestamp'])

# # --------Round-Trip Time [ms]-----------
df_exploration = df_exploration.drop(columns=['Round-Trip Time [ms]'])

# # --------IP Address-----------
df_exploration = df_exploration.drop(columns=['IP Address'])

# # --------Region-----------
df_exploration['Region'] = df_exploration['Region'].fillna('Unknown')

# # --------City-----------
df_exploration['City'] = df_exploration['City'].fillna('Unknown')

# --------User Agent String-----------
df_exploration = df_exploration.drop(columns=['User Agent String'])

# --------Browser Name and Version--------
# # Afficher les valeurs uniques
# # print(df['Browser Name and Version'].value_counts())
# # Split in 2
# # Utiliser str.extract avec une expression régulière pour séparer le nom et la versiondf_exploration[['Browser Name', 'Browser Version']] = df_exploration['Browser Name and Version'].str.extract(r'([\w\s]+)(?:\s([\d\.]+))?')
# print(df_exploration.loc[41282, 'Browser Name and Version'])

df_exploration[['Browser Name', 'Browser Version']] = df_exploration['Browser Name and Version'].str.extract(r'(^[\D]+)(\d+[\d\.]*)?')

# Décicion price de réunir les nom d'OS
# Exemple de regroupement pour Chrome, Edge et Safari
conditions = [
    df_exploration['Browser Name'].str.contains('Chrome', case=False, na=False),
    df_exploration['Browser Name'].str.contains('Edge', case=False, na=False),
    df_exploration['Browser Name'].str.contains('Safari', case=False, na=False),
    df_exploration['Browser Name'].str.contains('Firefox', case=False, na=False),
    df_exploration['Browser Name'].str.contains('Opera', case=False, na=False)
]

# Les choix correspondent aux conditions
choices = ['Chrome', 'Edge', 'Safari', 'Firefox', 'Opera']

# Appliquer les conditions pour mettre à jour seulement les valeurs correspondantes
for condition, choice in zip(conditions, choices):
    df_exploration.loc[condition, 'Browser Name'] = choice

# Ajouter la condition pour les Bots et l'appliquer
condition_bot = df_exploration['Browser Name'].str.contains('Bot|bot', case=False, na=False)
df_exploration.loc[condition_bot, 'Browser Name'] = 'Bot'

df_exploration['Browser Name'] = df_exploration['Browser Name'].fillna('Unknown')
df_exploration['Browser Version'] = df_exploration['Browser Version'].fillna('Unknown')

df_exploration = df_exploration.drop(columns=['Browser Name and Version'])

# --------OS Name and Version-----------

# Afficher les valeurs uniques
# print(df['OS Name and Version'].value_counts())
# # Split in 2
# # Utiliser str.extract avec une expression régulière pour séparer le nom et la version

df_exploration[['OS Name', 'OS Version']] = df_exploration['OS Name and Version'].str.extract(r'(^[\D]+)(\d+[\d\.]*)?')
df_exploration['OS Name'] = df_exploration['OS Name'].fillna('Unknown')
df_exploration['OS Version'] = df_exploration['OS Version'].fillna('Unknown')

df_exploration = df_exploration.drop(columns=['OS Name and Version'])

# --------Device Type-----------
df_exploration['Device Type'] = df_exploration['Device Type'].fillna('Unknown')

# -------fin transforme aperçu
print(df_exploration.isnull().sum())


# --------Browser Name and Version--------
# # pd.set_option('display.max_rows', None)
# print(df['Browser Name'].value_counts())
# takeover_df = df[df['Is Account Takeover'] == True]

# # Compter le nombre de takeovers par groupe de navigateur
# takeover_counts = takeover_df['Browser Name'].value_counts()

# # Afficher les résultats
# print(takeover_counts)
# Création de la table de contingence
# contingency_table = pd.crosstab(df['Browser Name'], df['Is Account Takeover'])

# # Afficher la table de contingence
# print(contingency_table)

# # Effectuer le test du Chi-carré
# chi2, p, dof, expected = chi2_contingency(contingency_table)

# print(f"Chi2 Stat: {chi2}")
# print(f"P-value: {p}")

# Afficher les résultats
# pd.set_option('display.max_rows', None)
# print(df['Browser Name'].value_counts())

# --------colonne OS Name and Version-----------
# Afficher les valeurs uniques
# print(df['OS Name and Version'].value_counts())
# # Split in 2
# # Utiliser str.extract avec une expression régulière pour séparer le nom et la version
# df[['OS Name', 'OS Version']] = df['OS Name and Version'].str.extract(r'(^[\D]+)(\d+[\d\.]*)')

# print(df['OS Name'].value_counts())


# """""""""""""""colonne User Agent String
# Exemple d'User Agent String
# ua_string = 'Mozilla/5.0 (Linux; U; Android 13.0; i phone X Build/NRD90M; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/85.0.4183.127 Mobile Safari/537.36 OPR/52.1.2254.54298'

# # Parse le User Agent String
# user_agent = parse(ua_string)
# print(user_agent)
# # Accéder aux propriétés
# print('Navigateur :', user_agent.browser.family)  # Navigateur
# print('Version du Navigateur :', user_agent.browser.version_string)  # Version du Navigateur
# print('OS :', user_agent.os.family)  # OS
# print('Version de l\'OS :', user_agent.os.version_string)  # Version de l'OS
# print('Mobile :', user_agent.is_mobile)  # Mobile ou non


# -------------------visualisation
# # Exemple de Code pour un Barplot (Connexions par Pays)
# # Préparer les données (exemple avec Country, remplacez par vos données réelles)
# country_counts = df['Country'].value_counts().reset_index().rename(columns={'index': 'Pays', 'count': 'Connections'})
# # Afficher seulement les 10 premiers pays par nombre de connexions
# top_countries = country_counts.head(20)
# # Créer un barplot avec Seaborn 
# plt.figure(figsize=(10, 8))
# sns.barplot(data=top_countries, x='Connections', y='Country')
# plt.title('Nombre de Connexions par Pays')
# plt.xlabel('Nombre de Connexions')
# plt.ylabel('Pays')
# plt.show()

#  1. Évolution des Connexions dans le Temps
# Extraire l'heure de chaque timestamp
# # Convertir la colonne 'Login Timestamp' en datetime
# df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
# df['Hour'] = df['Login Timestamp'].dt.hour

# # Compter le nombre de connexions pour chaque heure
# hourly_connections = df.groupby('Hour').size()

# # Créer le graphique
# plt.figure(figsize=(12, 6))
# plt.bar(hourly_connections.index, hourly_connections, width=0.8, color='skyblue')

# plt.title('Nombre de Connexions par Heure sur une Journée Typique')
# plt.xlabel('Heure de la Journée')
# plt.ylabel('Nombre de Connexions')
# plt.xticks(hourly_connections.index)
# plt.grid(axis='y')

# plt.show()

#  6. Utilisation des Navigateurs et des Systèmes d'Exploitation
# Calculer le nombre de connexions pour chaque nom de navigateur
# browser_name_counts = df['Browser Name'].value_counts().reset_index()
# browser_name_counts.columns = ['Browser', 'Count']

# # Créer un diagramme à barres pour les noms des navigateurs
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Count', y='Browser', data=browser_name_counts, palette='viridis')
# plt.title('Utilisation des Navigateurs')
# plt.xlabel('Nombre de Connexions')
# plt.ylabel('')
# plt.tight_layout()
# plt.show()

# Se concentrer sur le top N navigateurs et regrouper les autres
# top_n = 5
# top_browsers = browser_name_counts.head(top_n)
# other_browsers_sum = browser_name_counts['Count'][top_n:].sum()
# top_browsers = top_browsers._append({'Browser': 'Autres', 'Count': other_browsers_sum}, ignore_index=True)

# # Créer un diagramme circulaire avec les navigateurs regroupés
# plt.figure(figsize=(8, 8))
# plt.pie(top_browsers['Count'], labels=top_browsers['Browser'], autopct='%1.1f%%', startangle=140)
# plt.title('Répartition des Principaux Navigateurs et Autres')
# plt.show()

# 1. Distribution des Heures de Connexion
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='Hour', bins=24, kde=True)
# plt.title('Distribution des Heures de Connexion')
# plt.xlabel('Heure')
# plt.ylabel('Nombre de Connexions')
# plt.xticks(range(0, 24))
# plt.show()


# boxplots pour les heures de connexion (hours) segmentées par is Weekend ou par Device type.
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Device Type', y='Hour', data=df)
# plt.title('Heures de Connexion par Type d\'Appareil')
# plt.xlabel('Type d\'Appareil')
# plt.ylabel('Heure')
# plt.show()


# <<<<<----- preparation de la data

# doublons = df_exploration.duplicated()
# print(doublons.sum())  # Affiche le nombre total de doublons
# il y a aucun doublons

# SUITE  DE QUE FAIRE COLONNE PAR COLONNE encodage
# <<<<<<<<<<<<------binary encoding for 'Country', 'Region', 'City', 'ASN'
print("Types de colonnes avant transformations :")
print(df_exploration.dtypes)
# Étape 1: Encodage Binaire
cols_to_encode_binary = ['Country', 'Region', 'City', 'ASN', 'Browser Name', 'Browser Version', 'OS Name', 'OS Version']
encoder_binary = ce.BinaryEncoder(cols=cols_to_encode_binary)
df_encoded_binary = encoder_binary.fit_transform(df_exploration)

# Étape 2: One-Hot Encoding sur le DataFrame résultant de l'encodage binaire
cols_to_encode_onehot = ['Device Type', 'Day', 'Hour']
df_encoded = pd.get_dummies(df_encoded_binary, columns=cols_to_encode_onehot)

# Étape 3: Conversion des colonnes booléennes
bool_cols = ['Login Successful', 'Is Attack IP', 'Is Account Takeover', 'Is Weekend', 'Outside Office Hours']
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# # Afficher les types de colonnes après toutes les transformations
# print("Types de colonnes après transformations :")
# pd.set_option('display.max_rows', None)
# print(df_encoded.dtypes)

# # /tester
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

X = df_encoded.drop('Is Account Takeover', axis=1)
y = df_encoded['Is Account Takeover']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Appliquer SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)



# # # Exemple avec un classificateur Random Forest
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Exactitude du modèle : {accuracy:.2f}")

model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitude du modèle : {accuracy:.2f}")

# # Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion avec Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.ylabel('Vérité')
plt.xlabel('Prédiction')
plt.title('Matrice de Confusion')
plt.show()