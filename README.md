# 📊 Churn Prediction Dashboard with MLOps

A machine learning dashboard to predict customer churn using Streamlit, MLflow, XGBoost, and Docker. Built with MLOps best practices and deployed for real-time inference.

## 🚀 Features
- Real-time churn prediction UI
- MLflow-powered experiment tracking
- Dockerized app with production-ready setup
- Optional database integration

## 📦 Stack
- **Frontend**: Streamlit
- **Modeling**: Scikit-learn, XGBoost
- **MLOps**: MLflow, Docker, GitHub Actions (optional)
- **Data**: Telco Customer Churn Dataset
# 📊 Churn Prediction Dashboard with MLOps

A machine learning dashboard to predict customer churn based on the Telco dataset. This project showcases model training, hyperparameter tuning, deployment, and a fully interactive Streamlit frontend.

🔗 **Live Demo:** [\[Insert your Streamlit Cloud link here\]](https://ml-churn-dashboard.streamlit.app/)

## 🚀 Features
- Real-time churn prediction UI with Streamlit
- Simulate high-risk and loyal customer scenarios
- Tuned XGBoost model with GridSearchCV
- MLflow-powered experiment tracking
- Dockerized for portability
- Clean modular file structure

## 🧠 Tech Stack
- **Frontend**: Streamlit
- **Modeling**: Scikit-learn, XGBoost
- **Experiment Tracking**: MLflow
- **Automation**: GitHub, Docker
- **Data**: Telco Customer Churn (Kaggle)

## 📁 Folder Structure
```
ml-churn-dashboard/
├── app/                  # Streamlit app code
│   ├── dashboard.py
│   └── preprocess.py
├── data/                 # Dataset and processed CSV
├── models/               # Saved XGBoost model + scaler
├── plots/                # Confusion matrix, ROC, feature importances
├── requirements.txt
├── README.md
```

## 🛠️ Run Locally

```bash
git clone https://github.com/yourusername/ml-churn-dashboard.git
cd ml-churn-dashboard
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## 🐳 Run with Docker

```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

## 📝 Notes
- Supports manual input and preset simulations
- Compatible with Streamlit Cloud and Hugging Face Spaces
- Great for demoing ML, MLOps, and deployment skills