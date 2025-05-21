import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from xgboost import plot_importance

def plot_logreg_importance(model, feature_names):
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefs
    }).sort_values(by = 'Importance', key=abs, ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=coef_df.head(10))
    plt.title("Top 10 Feature Importance - Logistic Regression")
    plt.tight_layout()
    plt.savefig("plots/logreg_importance.png")
    plt.close


def plot_xgb_importance(model):
    ax = plot_importance(model, max_num_features=10)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig("plots/xgb_importance.png")
    plt.close()
