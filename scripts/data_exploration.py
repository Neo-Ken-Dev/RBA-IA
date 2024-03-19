import pandas as pd
import matplotlib.pyplot as plt

df_exploration = pd.read_csv('rba-dataset-complet.csv')

# ---- 3) Vérification de la data--------------
# 1 Première visualisation :
print(df_exploration.head(10))

# 2 Nom et type des colonnes :
# description rapide de la data
print(df_exploration.info())

# 3/ Compréhension des entrées null dans le dataset.
print(df_exploration.isnull().sum())

# 3.3.3.1 Round-trip time
print(df_exploration.groupby('Device Type')['Round-Trip Time [ms]'].count())

# 3.3.3.2 Les Régions
null_regions = df_exploration[df_exploration['Region'].isnull()]
# Distribution des regions null par pays:
print(null_regions['Country'].value_counts())

# Distribution des regions null parville:
print(null_regions['City'].value_counts())

# ****Comparer le pourcentage de connexions réussies entre les données avec et sans région****
null_regions = df_exploration[df_exploration['Region'].isnull()]
total_logins_no_region = len(null_regions)
login_success_count_no_region = null_regions['Login Successful'].value_counts()
rate_success_no_region = (login_success_count_no_region[True] / total_logins_no_region) * 100
rate_failure_no_region = (login_success_count_no_region[False] / total_logins_no_region) * 100

not_null_regions = df_exploration[df_exploration['Region'].notnull()]
total_logins_with_region = len(not_null_regions)
login_success_count_with_region = not_null_regions['Login Successful'].value_counts()
rate_success_with_region = (login_success_count_with_region[True] / total_logins_with_region) * 100
rate_failure_with_region = (login_success_count_with_region[False] / total_logins_with_region) * 100

data = {
    "Condition": ["Sans Region", "Avec Region"],
    "Total Logins": [total_logins_no_region, total_logins_with_region],
    "Login Successful": [login_success_count_no_region[True], login_success_count_with_region[True]],
    "Login Failure": [login_success_count_no_region[False], login_success_count_with_region[False]],
    "Success Rate (%)": [rate_success_no_region, rate_success_with_region],
    "Failure Rate (%)": [rate_failure_no_region, rate_failure_with_region]
}

# Convert data to DataFrame
comparison_df_login_success = pd.DataFrame(data)
print(comparison_df_login_success)

# ****Comparer le pourcentage de connexions frauduleuse entre les données avec et sans région****
null_regions = df_exploration[df_exploration['Region'].isnull()]
total_logins_no_region = len(null_regions)
takeover_count_no_region = null_regions['Is Account Takeover'].value_counts()
rate_fraud_no_region = (takeover_count_no_region[True] / total_logins_no_region) * 100
rate_not_fraud_no_region = (takeover_count_no_region[False] / total_logins_no_region) * 100

not_null_regions = df_exploration[df_exploration['Region'].notnull()]
total_logins_with_region = len(not_null_regions)
takeover_count_with_region = not_null_regions['Is Account Takeover'].value_counts()
rate_fraud_with_region = (takeover_count_with_region[True] / total_logins_with_region) * 100
rate_not_fraud_with_region = (takeover_count_with_region[False] / total_logins_with_region) * 100

data = {
    "Condition": ["Sans Region", "Avec Region"],
    "Total Logins": [total_logins_no_region, total_logins_with_region],
    "Login Fraud": [takeover_count_no_region[True], takeover_count_with_region[True]],
    "Login Not Fraud": [takeover_count_no_region[False], takeover_count_with_region[False]],
    "Fraud Rate (%)": [rate_fraud_no_region, rate_fraud_with_region],
    "Not Fraud Rate (%)": [rate_not_fraud_no_region, rate_not_fraud_with_region]
}

# Convert data to DataFrame
comparison_df_login_fraud = pd.DataFrame(data)
print(comparison_df_login_fraud)

# Calculer le Ratio
if rate_fraud_with_region > 0:  # Pour éviter la division par zéro
    ratio = rate_fraud_no_region / rate_fraud_with_region
    print(f"Ratio des pourcentages (sans région / avec région) : {ratio}")
else:
    print("Le ratio ne peut pas être calculé car le dénominateur est zéro.")

# Visualisation avec matplotlib
plt.bar(['Avec région', 'Sans région'], [rate_fraud_with_region, rate_fraud_no_region])
plt.ylabel('Pourcentage de Is Account Takeover à True')
plt.title('Comparaison des pourcentages de Fraude avec et sans region')
plt.show()


# 3.3.3.3. Les villes null
null_cities = df_exploration[df_exploration['City'].isnull()]

# Distribution des villes null par pays:
print(null_cities['Country'].value_counts())

# Distribution des villes null par région:
print(null_cities['Region'].value_counts())

# Distribution des villes null par type d'appareil:
print(null_cities['Device Type'].value_counts())

print((null_cities['Is Account Takeover'] == True).sum())

# 3.3.3.4 Les type d’appareil null
null_devices = df_exploration[df_exploration['Device Type'].isnull()]
print((null_devices['Is Account Takeover'] == True).sum())

# 3.3.4.	 Les entrées de type Object :
# 3.3.4.1 Colonne “Browser Name and Version”
print(df_exploration['Browser Name and Version'].value_counts())

# 3.3.4.2 Colonne “Os Name and Version”
print(df_exploration['OS Name and Version'].value_counts())