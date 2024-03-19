import pandas as pd
import category_encoders as ce

def encode_data(df):
    # 3.5.1 Encodage Binaire
    cols_to_encode_binary = ['Country', 'Region', 'City', 'ASN', 'Browser Name', 'Browser Version', 'OS Name', 'OS Version']
    encoder_binary = ce.BinaryEncoder(cols=cols_to_encode_binary)
    df_encoded_binary = encoder_binary.fit_transform(df)

    # 3.5.2 One-Hot Encoding
    # One-Hot Encoding sur le DataFrame résultant de l'encodage binaire
    cols_to_encode_onehot = ['Device Type', 'Day', 'Hour']
    df_encoded = pd.get_dummies(df_encoded_binary, columns=cols_to_encode_onehot)

    # Conversion des booléens
    bool_cols = ['Login Successful', 'Is Attack IP', 'Is Account Takeover', 'Is Weekend', 'Outside Office Hours']
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    return df_encoded

# df_encoded.to_csv('data_encoded.csv', index=False)
