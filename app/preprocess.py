import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load reference columns from training data
training_data_path = "data/preprocessed_churn.csv"
reference_cols = pd.read_csv(training_data_path).drop("Churn", axis=1).columns.tolist()

# Fit scaler on training distribution for consistency
def fit_scaler():
    df = pd.read_csv(training_data_path)
    scaler = joblib.load("models/scaler.pkl")
    return scaler

scaler = fit_scaler()

def preprocess_input(user_df):
    # Drop customerID if exists
    if "customerID" in user_df.columns:
        user_df = user_df.drop("customerID", axis=1)

    # Fix TotalCharges to numeric
    user_df['TotalCharges'] = pd.to_numeric(user_df['TotalCharges'], errors='coerce')

    # Handle missing TotalCharges
    user_df['TotalCharges'] = user_df['TotalCharges'].fillna(user_df['TotalCharges'].median())

    # Map churn if included (for test cases)
    if "Churn" in user_df.columns:
        user_df["Churn"] = user_df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode all categoricals with drop_first=True
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']
    
    user_df = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)

    # Add missing columns from training
    for col in reference_cols:
        if col not in user_df.columns:
            user_df[col] = 0

    # Reorder
    user_df = user_df[reference_cols]

    # Standard scale numeric columns
    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
    user_df[numeric] = scaler.transform(user_df[numeric])

    return user_df