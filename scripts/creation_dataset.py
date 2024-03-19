import pandas as pd

# ---- 2) Création d’une copie de la data--------------
# Lecture du dataset
df = pd.read_csv('rba-dataset-complet.csv')
# <<<<< Modification 2:  aprés 4.2/ Premier test:
print("Nombre de connexions frauduleuses:", df['Is Account Takeover'].sum())
# >>>>>>

# <<<<< Modification 1:  avant 3.4/ Préparation de la data:
# Réduction du Dataset à 1 000 000 de lignes 
# df_to_save = df[0:1000000]
# df_to_save.to_csv('rba-smallDataset.csv')
df = pd.read_csv('rba-smallDataset.csv')
# >>>>>>

# <<<<< Modification 2:  aprés 4.2/ Premier test:
# création d'un petit dataset avec tous les Is Account takeover et 100 000 non takeover
# Séparer les instances où Is Account Takeover est True
takeover_true = df[df['Is Account Takeover'] == True]

# Séparer les instances où Is Account Takeover est False
takeover_false = df[df['Is Account Takeover'] == False]

# Échantillonnage aléatoire de 100 000 d'instances non takeover
takeover_false_sample = takeover_false.sample(n=100000, random_state=42)

# Concaténer les instances True et l'échantillon aléatoire des instances False
new_dataset = pd.concat([takeover_true, takeover_false_sample])

# Mélanger les lignes du nouveau dataset pour éviter toute séquence
new_dataset = new_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

new_dataset.to_csv('rba-smallDataset.csv')
print("Le nouveau dataset réduit contenant toutes les connexions frauduleuses a été créé et sauvegardé.")
# >>>>>>
