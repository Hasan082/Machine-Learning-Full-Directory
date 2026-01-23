import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


def clean_data(df):
    print("🧹 Cleaning data...")
    df = handle_missing_values(df)
    df = df.drop_duplicates()
    print(f"✅ Cleaned shape: {df.shape}")
    return df


def encode_and_scale(df, target_column):
    label_encoders = {}
    scaler = StandardScaler()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        label_encoders['target'] = le_target

    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, scaler, label_encoders
