import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import *


def evaluate_model(
        model,
        X_test,
        y_test,
        feature_cols):

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:",
          accuracy_score(y_test, pred))

    print("Precision:",
          precision_score(y_test, pred))

    print("Recall:",
          recall_score(y_test, pred))

    print("F1:",
          f1_score(y_test, pred))

    print("ROC-AUC:",
          roc_auc_score(y_test, prob))

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title("Confusion Matrix")
    plt.show()

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance":
        model.feature_importances_
    })

    print("\nFeature Importance:")
    print(importance)

    return pred, prob


