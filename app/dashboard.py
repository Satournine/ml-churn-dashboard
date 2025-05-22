import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_input
from PIL import Image

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
model = joblib.load("models/best_model.pkl")

st.sidebar.title("Telco Churn Predictor")
st.sidebar.markdown("Fill out customer information below: ")

def user_input_form():
    if "preset_applied" not in st.session_state:
        st.session_state.preset_applied = False

    def apply_churner_preset():
        st.session_state.gender = "Female"
        st.session_state.SeniorCitizen = 1
        st.session_state.Partner = "No"
        st.session_state.Dependents = "No"
        st.session_state.tenure = 1
        st.session_state.PhoneService = "Yes"
        st.session_state.MultipleLines = "No"
        st.session_state.InternetService = "Fiber optic"
        st.session_state.OnlineSecurity = "No"
        st.session_state.OnlineBackup = "No"
        st.session_state.DeviceProtection = "No"
        st.session_state.TechSupport = "No"
        st.session_state.StreamingTV = "Yes"
        st.session_state.StreamingMovies = "Yes"
        st.session_state.Contract = "Month-to-month"
        st.session_state.PaperlessBilling = "Yes"
        st.session_state.PaymentMethod = "Electronic check"
        st.session_state.MonthlyCharges = 95.0
        st.session_state.TotalCharges = 100.0
        st.session_state.preset_applied = True
    def apply_loyal_preset():
        st.session_state.gender = "Male"
        st.session_state.SeniorCitizen = 0
        st.session_state.Partner = "Yes"
        st.session_state.Dependents = "Yes"
        st.session_state.tenure = 60
        st.session_state.PhoneService = "Yes"
        st.session_state.MultipleLines = "Yes"
        st.session_state.InternetService = "DSL"
        st.session_state.OnlineSecurity = "Yes"
        st.session_state.OnlineBackup = "Yes"
        st.session_state.DeviceProtection = "Yes"
        st.session_state.TechSupport = "Yes"
        st.session_state.StreamingTV = "Yes"
        st.session_state.StreamingMovies = "Yes"
        st.session_state.Contract = "Two year"
        st.session_state.PaperlessBilling = "No"
        st.session_state.PaymentMethod = "Credit card (automatic)"
        st.session_state.MonthlyCharges = 30.0
        st.session_state.TotalCharges = 2500.0
        st.session_state.preset_applied = True

    if st.sidebar.button("Simulate Likely Churner"):
        apply_churner_preset()
    if st.sidebar.button("Simulate Loyal Customer"):
        apply_loyal_preset()

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="gender")
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1], key="SeniorCitizen")
    Partner = st.sidebar.selectbox("Partner", ["Yes", "No"], key="Partner")
    Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"], key="Dependents")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12, key="tenure")
    PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"], key="PhoneService")
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="MultipleLines")
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="InternetService")
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"], key="OnlineSecurity")
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="OnlineBackup")
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="DeviceProtection")
    TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="TechSupport")
    StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="StreamingTV")
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="StreamingMovies")
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="Contract")
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"], key="PaperlessBilling")
    PaymentMethod = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], key="PaymentMethod")
    MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 150.0, 70.0, key="MonthlyCharges")
    TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2500.0, key="TotalCharges")

    data = {
        "gender": st.session_state.gender,
        "SeniorCitizen": st.session_state.SeniorCitizen,
        "Partner": st.session_state.Partner,
        "Dependents": st.session_state.Dependents,
        "tenure": st.session_state.tenure,
        "PhoneService": st.session_state.PhoneService,
        "MultipleLines": st.session_state.MultipleLines,
        "InternetService": st.session_state.InternetService,
        "OnlineSecurity": st.session_state.OnlineSecurity,
        "OnlineBackup": st.session_state.OnlineBackup,
        "DeviceProtection": st.session_state.DeviceProtection,
        "TechSupport": st.session_state.TechSupport,
        "StreamingTV": st.session_state.StreamingTV,
        "StreamingMovies": st.session_state.StreamingMovies,
        "Contract": st.session_state.Contract,
        "PaperlessBilling": st.session_state.PaperlessBilling,
        "PaymentMethod": st.session_state.PaymentMethod,
        "MonthlyCharges": st.session_state.MonthlyCharges,
        "TotalCharges": st.session_state.TotalCharges
    }

    return pd.DataFrame([data])

input_df = user_input_form()
processed_input = preprocess_input(input_df)
prediction = model.predict(processed_input)[0]
probability = model.predict_proba(processed_input)[0][1]

st.title("Telco Customer Churn Prediction Dashboard")

st.markdown("""
    This interactive dashboard allows you to simulate Telco customer profiles and predict churn probability using a tuned XGBoost model.
            """)

tab1, tab2, tab3 = st.tabs(["Prediction", "Visualizations", "Model Info"])

with tab1:
    st.header("Prediction Result")
    st.write(f"**Prediction:** {'Churn' if prediction == 1 else 'No Churn'}")
    st.write(f"**Churn Probability:** {probability:.2f}")
    
    if prediction == 1:
        st.warning("This customer is likely to churn")
    else:
        st.warning("This customer is likely to stay.")

with tab2:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.image("plots/confusion_xgb_tuned.png", caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image("plots/roc_xgb_tuned.png", caption="ROC Curve", use_container_width=True)

with tab3:
    st.header("Top Features Driving Churn")
    st.image("plots/xgb_importance.png", caption="XGBoost Feature Importance", use_container_width=True)

