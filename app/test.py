import pandas as pd
import joblib
from preprocess import preprocess_input

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

for label, customer in [("HIGH-RISK", high_risk_input), ("LOW-RISK", low_risk_input)]:
    print(f"\n--- {label} CUSTOMER ---")
    X = preprocess_input(customer)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    print("Prediction:", "Churn" if pred == 1 else "No Churn")
    print("Churn Probability:", round(proba, 4))