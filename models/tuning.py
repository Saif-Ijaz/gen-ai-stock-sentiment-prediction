from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def tune_xgb(X, y):
    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    }

    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_
