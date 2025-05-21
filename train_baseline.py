import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

from app.plots import plot_logreg_importance, plot_xgb_importance
from app.plot_metrics import plot_confusion, plot_roc

df = pd.read_csv("data/preprocessed_churn.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_experiment("Churn Prediction")

with mlflow.start_run(run_name="Baseline_Logreg"):
    logreg = LogisticRegression(max_iter = 1000)
    logreg.fit(X_train, y_train)
    plot_logreg_importance(logreg, X.columns)
    y_pred_logreg = logreg.predict(X_test)

    acc = accuracy_score(y_test, y_pred_logreg)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(logreg, "logreg_model")

    print("Logistic Regression Accuracy: ", acc)
    print(classification_report(y_test, y_pred_logreg))

    y_proba_logreg = logreg.predict_proba(X_test)[:, 1]
    plot_confusion(y_test, y_pred_logreg, model_name="logreg")
    plot_roc(y_test, y_proba_logreg, model_name="logreg")

with mlflow.start_run(run_name="Baseline_XGBoost"):
    xgb = XGBClassifier(use_label_encoder = False, eval_metric = "logloss")
    xgb.fit(X_train, y_train)
    plot_xgb_importance(xgb)
    y_pred_xgb = xgb.predict(X_test)

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    mlflow.log_metric("accuracy", acc_xgb)
    mlflow.sklearn.log_model(xgb, "xgb_model")

    print("XGBoost Accuracy: ", acc_xgb)
    print(classification_report(y_test, y_pred_xgb))

    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    plot_confusion(y_test, y_pred_xgb, model_name="xgb")
    plot_roc(y_test, y_proba_xgb, model_name="xgb")