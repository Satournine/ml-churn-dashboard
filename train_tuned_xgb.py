import mlflow.sklearn
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
from app.plot_metrics import plot_confusion, plot_roc
from app.plots import plot_xgb_importance
import joblib


df = pd.read_csv("data/preprocessed_churn.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

mlflow.set_experiment("Churn Prediction")

with mlflow.start_run(run_name="Tuned_XGBoost"):
    param_grid = {
        "max_depth": [3,5,7],
        "learning_rate": [0.01, 0.1],
        "n_estimator": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=1,
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    mlflow.log_params(grid.best_params_)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc_score)
    mlflow.sklearn.log_model(best_model, "tuned_xgb_model")

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    print("ROC AUC:", auc_score)
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, "models/best_model.pkl")

    # Plots
    plot_confusion(y_test, y_pred, model_name="xgb_tuned")
    plot_roc(y_test, y_proba, model_name="xgb_tuned")
    plot_xgb_importance(best_model)