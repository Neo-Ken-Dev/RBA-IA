import pandas as pd
from sklearn.model_selection import train_test_split
import data_encoding as de

df = pd.read_csv('rba-smallDataset.csv')

# isoler un dataset d’entrainement et de test
# train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
# <<<<< Modification 2:  aprés 4.2/ Premier test:
train_set, test_set = train_test_split(df, test_size=0.2, stratify=df['Is Account Takeover'], random_state=42)
# >>>>>>

df_to_preprocess = train_set.copy()


# 3.4/ Préparation de la data
# 3.4.1. Colonne Index
# Création d'un méthode pour supprimer les colonnes spécifiées d'un DataFrame.
def drop_columns(dataframe, columns_to_drop):
    """
    Supprime les colonnes spécifiées d'un DataFrame.

    Paramètres :
    dataframe (pd.DataFrame) : Le DataFrame à partir duquel supprimer les colonnes.
    columns_to_drop (list) : Liste des noms de colonnes à supprimer.

    Retourne :
    pd.DataFrame : Un nouveau DataFrame sans les colonnes spécifiées.
    """
    return dataframe.drop(columns=columns_to_drop)
# Drop de la colonne Index:
df_to_preprocess = drop_columns(df_to_preprocess, ['index'])

# 3.4.2. Colonne Login Timestamp
# Convertir la colonne 'Login Timestamp' en datetime
df_to_preprocess['Login Timestamp'] = pd.to_datetime(df_to_preprocess['Login Timestamp'])

# Méthodes pour extraire les informations de mon Timestamp
def extract_day_and_hour(df, column_name):
    """
    Extrait le jour et l'heure d'une colonne datetime et les ajoute en tant que nouvelles colonnes.
    """
    df['Day'] = df[column_name].dt.day
    df['Hour'] = df[column_name].dt.hour
    return df

def identify_weekend(df, column_name):
    """
    Identifie si la date est pendant le weekend.
    """
    df['Is Weekend'] = df[column_name].dt.dayofweek >= 5
    return df

def outside_office_hours(df, hour_column):
    """
    Détermine si l'heure est en dehors des heures de bureau (avant 7h ou après 21h).
    """
    df['Outside Office Hours'] = (df[hour_column] < 7) | (df[hour_column] > 21)
    return df

# Appliquez chaque transformation en utilisant les méthodes définies
timestamp_column = 'Login Timestamp'
df_to_preprocess = extract_day_and_hour(df_to_preprocess, timestamp_column)
df_to_preprocess = identify_weekend(df_to_preprocess, timestamp_column)
df_to_preprocess = outside_office_hours(df_to_preprocess, 'Hour')

df_to_preprocess = drop_columns(df_to_preprocess, ['Login Timestamp'])

# 3.4.3. Colonne User ID
df_to_preprocess = drop_columns(df_to_preprocess, ['User ID'])

# 3.4.4. Colonne Round-Trip Time
df_to_preprocess = drop_columns(df_to_preprocess, ['Round-Trip Time [ms]'])

# 3.4.5. Colonne IP Address
df_to_preprocess = drop_columns(df_to_preprocess, ['IP Address'])

# 3.4.6. Colonne Region
def fill_missing_values(df, column_name, fill_value):
    """
    Remplit les valeurs manquantes d'une colonne spécifiée avec une valeur donnée.

    Paramètres :
    df (pd.DataFrame) : DataFrame sur lequel appliquer la transformation.
    column_name (str) : Nom de la colonne à traiter.
    fill_value : Valeur à utiliser pour remplir les valeurs manquantes.

    Retourne :
    pd.DataFrame : DataFrame avec les valeurs manquantes remplies.
    """
    df[column_name] = df[column_name].fillna(fill_value)
    return df

df_to_preprocess = fill_missing_values(df_to_preprocess, 'Region', 'Unknown')

# 3.4.7. Colonne City
def drop_rows_with_nulls(df, column_name):
    """
    Supprime les lignes contenant des valeurs null dans la colonne spécifiée.

    Paramètres :
    df (pd.DataFrame) : Le DataFrame à traiter.
    column_name (str) : Le nom de la colonne pour laquelle supprimer les lignes avec des valeurs null.

    Retourne :
    pd.DataFrame : Un nouveau DataFrame sans les lignes contenant des valeurs null dans la colonne spécifiée.
    """
    return df.dropna(subset=[column_name])

df_to_preprocess = drop_rows_with_nulls(df_to_preprocess, 'City')

# 3.4.8. Colonne User Agent String
df_to_preprocess = drop_columns(df_to_preprocess, ['User Agent String'])

# 3.4.9. Colonne Browser Name And Version
# Extraire le nom de la verion
def extract_info(df, column_name, new_col_name1, new_col_name2):
    """
    Extrait des informations basées sur une expression régulière et les ajoute comme nouvelles colonnes.

    Paramètres :
    df (pd.DataFrame) : DataFrame contenant les données.
    column_name (str) : Nom de la colonne contenant les informations à extraire.
    new_col_name1 (str) : Nom de la première nouvelle colonne pour les données extraites.
    new_col_name2 (str) : Nom de la deuxième nouvelle colonne pour les données extraites.

    Retourne :
    pd.DataFrame : DataFrame modifié avec deux nouvelles colonnes pour les informations extraites.
    """
    df[[new_col_name1, new_col_name2]] = df[column_name].str.extract(r'(^[\D]+)(\d+[\d\.]*)?')

    return df

df_to_preprocess = extract_info(df_to_preprocess, 'Browser Name and Version', 'Browser Name', 'Browser Version')

# Réunir les nom des navigateur
conditions = [
    df_to_preprocess['Browser Name'].str.contains('Chrome', case=False, na=False),
    df_to_preprocess['Browser Name'].str.contains('Edge', case=False, na=False),
    df_to_preprocess['Browser Name'].str.contains('Safari', case=False, na=False),
    df_to_preprocess['Browser Name'].str.contains('Firefox', case=False, na=False),
    df_to_preprocess['Browser Name'].str.contains('Opera', case=False, na=False)
]
# Appliquer les conditions pour mettre à jour seulement les valeurs correspondantes
choices = ['Chrome', 'Edge', 'Safari', 'Firefox', 'Opera']
for condition, choice in zip(conditions, choices):
    df_to_preprocess.loc[condition, 'Browser Name'] = choice

# Ajouter la condition pour les Bots et l'appliquer
condition_bot = df_to_preprocess['Browser Name'].str.contains('Bot|bot', case=False, na=False)
df_to_preprocess.loc[condition_bot, 'Browser Name'] = 'Bot'

# Completer les valeurs qui sont à null si besoin
df_to_preprocess = fill_missing_values(df_to_preprocess, 'Browser Name', 'Unknown')
df_to_preprocess = fill_missing_values(df_to_preprocess, 'Browser Version', 'Unknown')

# Supprimer la colonne plus utile
df_to_preprocess = drop_columns(df_to_preprocess, ['Browser Name and Version'])

# 3.4.10. Colonne OS Name And Version
df_to_preprocess = extract_info(df_to_preprocess, 'OS Name and Version', 'OS Name', 'OS Version')
df_to_preprocess = fill_missing_values(df_to_preprocess, 'OS Name', 'Unknown')
df_to_preprocess = fill_missing_values(df_to_preprocess, 'OS Version', 'Unknown')
df_to_preprocess = drop_columns(df_to_preprocess, ['OS Name and Version'])

# 3.4.11. Device Type
df_to_preprocess = drop_rows_with_nulls(df_to_preprocess, 'Device Type')

# 3.4.12 gestion des doublons
# Affiche le nombre total de doublons
doublons = df_to_preprocess.duplicated()
print(doublons.sum())

# Encodage de la data
df_encoded = de.encode_data(df_to_preprocess)
df_encoded.to_csv('data_encoded.csv')



