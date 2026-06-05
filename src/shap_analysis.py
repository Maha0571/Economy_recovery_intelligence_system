import shap
import pandas as pd

from xgboost import XGBClassifier


def run_shap_analysis(
        model,
        X_test):

    X_sample = X_test.sample(
        min(300, len(X_test)),
        random_state=42
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_sample)

    shap.summary_plot(
        shap_values.values,
        X_sample
    )


