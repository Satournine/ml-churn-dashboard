import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from app.plot_metrics import plot_confusion, plot_roc
from app.plots import plot_xgb_importance

df = pd.read_csv("data/preprocessed_churn.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

mlflow.set_experiment("Churn Prediction")

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)


best_params = study.best_params
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]


with mlflow.start_run(run_name="Optuna_XGBoost"):
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))
    mlflow.sklearn.log_model(best_model, "optuna_xgb_model")

    print("Best Params:", best_params)
    print(classification_report(y_test, y_pred))

    plot_confusion(y_test, y_pred, model_name="xgb_optuna")
    plot_roc(y_test, y_proba, model_name="xgb_optuna")
    plot_xgb_importance(best_model)