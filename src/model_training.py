from xgboost import XGBClassifier


def train_model(X_train, y_train):

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.02,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


