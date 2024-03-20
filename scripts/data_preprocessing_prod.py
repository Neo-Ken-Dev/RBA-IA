import pandas as pd
from user_agents import parse

def extract_timestamp_features(df):
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
    df['Day'] = df['Login Timestamp'].dt.day
    df['Hour'] = df['Login Timestamp'].dt.hour
    df['Is Weekend'] = df['Login Timestamp'].dt.dayofweek >= 5
    df['Outside Office Hours'] = (df['Hour'] < 7) | (df['Hour'] > 21)
    df.drop(columns=['Login Timestamp'])
    return df

# Fonction pour extraire les informations du User Agent String
def extract_user_agent_info(ua_string):
    user_agent = parse(ua_string)
    browser_name = user_agent.browser.family
    browser_version = user_agent.browser.version_string
    os_name = user_agent.os.family
    os_version = user_agent.os.version_string
    device_type = get_device_type(user_agent)
    return browser_name, browser_version, os_name, os_version, device_type

def get_device_type(user_agent):
    if user_agent.is_mobile:
        return 'Mobile'
    elif user_agent.is_tablet:
        return 'Tablet'
    elif user_agent.is_pc:
        return 'PC'
    elif user_agent.is_bot:
        return 'Bot'
    else:
        return 'Other'

# Fonction qui sera utilisée pour transformer les informations reçu de l'API
def process_data(json_data):
    # Transformer JSON en DataFrame
    df = pd.DataFrame([json_data])
    df = df.copy()

    ua_string = df['User Agent String'].iloc[0]
    browser_name, browser_version, os_name, os_version, device_type = extract_user_agent_info(ua_string)

    # Ajouter les informations extraites au DataFrame original
    df['Browser Name'] = browser_name
    df['Browser Version'] = browser_version
    df['OS Name'] = os_name
    df['OS Version'] = os_version
    df['Device Type'] = device_type

    df.drop(columns=['User Agent String'])

    return df
