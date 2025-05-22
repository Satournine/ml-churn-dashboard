import pandas as pd
import joblib
from app.preprocess import preprocess_input

# Load model
model = joblib.load("models/best_model.pkl")

# High-risk churn customer
high_risk_input = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 150.0,
    "TotalCharges": 10000.0
}])

# Low-risk loyal customer
low_risk_input = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 30.0,
    "TotalCharges": 3000.0
}])

def test_high_risk_prediction():
    X = preprocess_input(high_risk_input)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    assert pred == 1 or proba > 0.5

def test_low_risk_prediction():
    X = preprocess_input(low_risk_input)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    assert pred == 0 or proba < 0.5