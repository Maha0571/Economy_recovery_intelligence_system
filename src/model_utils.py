import os
import joblib

def save_model(
        model,
        feature_cols):

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_dir = os.path.join(
        base_dir,
        "..",
        "models"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        model_dir,
        "economic_recovery_model.pkl"
    )

    feature_path = os.path.join(
        model_dir,
        "feature_columns.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        feature_cols,
        feature_path
    )


