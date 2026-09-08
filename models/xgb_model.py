# models/xgb_model.py
from xgboost import XGBClassifier

def build_xgb_model(scale_pos_weight=1.0):
    """
    Optimized XGBoost classifier with class balance handling.
    scale_pos_weight: ratio of negative/positive class to balance predictions
    """
    return XGBClassifier(
        n_estimators      = 500,   # was 300 — more trees = better generalization
        max_depth         = 4,     # was 6 — shallower trees reduce overfitting
        learning_rate     = 0.02,  # was 0.03 — slower learning = more precise
        subsample         = 0.7,   # was 0.8 — use 70% of rows per tree
        colsample_bytree  = 0.6,   # was 0.75 — use 60% of features per tree
        min_child_weight  = 5,     # was 3 — require more samples per leaf
        gamma             = 0.2,   # was 0.1 — higher pruning threshold
        reg_alpha         = 0.3,   # was 0.1 — stronger L1 regularization
        reg_lambda        = 2.0,   # was 1.5 — stronger L2 regularization
        eval_metric       = "logloss",
        scale_pos_weight  = scale_pos_weight,
        random_state      = 42,
        n_jobs            = -1
    )