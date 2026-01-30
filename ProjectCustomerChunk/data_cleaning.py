import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def handle_missing_values(df_missing_value):
    """Handle missing values in the DataFrame."""
    print("Handling missing values....")
    for col in df_missing_value.columns:
        if df_missing_value[col].isnull().sum() > 0:
            if df_missing_value[col].dtype in ['float64', 'int64']:  # Numerical Data
                df_missing_value[col].fillna(df_missing_value[col].mean(), inplace=True)
            else:
                df_missing_value[col].fillna(df_missing_value[col].mode()[0], inplace=True)
    print("Handling missing values done")
    return df_missing_value


def clean_data(df):
    """Clean the data by handling missing values and removing duplicates."""
    print("Clean data values....")
    df = handle_missing_values(df)
    df_dropduplicate = df.drop_duplicates()
    print(f"Data cleaned. Shape: {df_dropduplicate.shape}")
    return df_dropduplicate


def encode_data(df, target_column):
    """
    Encode categorical features and scale numerical features.
    
    Returns:
        X: Encoded and scaled features (DataFrame)
        y: Encoded target variable (array)
        scaler: Fitted StandardScaler
        label_encoders: Dictionary of fitted LabelEncoders for each categorical column
    """
    print("Encoding data values....")

    label_encoders = {}
    scaler = StandardScaler()

    # Make a safe copy and separate features from target
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    # Encode target if categorical
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        label_encoders[target_column] = le_target
        print(f"Target '{target_column}' encoded: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    # Identify column types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")

    # Encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded '{col}': {len(le.classes_)} unique values")

    # ✅ Scale ALL features (both original numeric and encoded categorical)
    # This ensures consistency between training and prediction
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    print(f"Scaled all {len(X.columns)} features")
    print("Encoding complete!")
    
    return X_scaled, y, scaler, label_encoders