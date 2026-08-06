# models/xgb_model.py
from xgboost import XGBClassifier

def build_xgb_model(scale_pos_weight=1.0):
    """
    Optimized XGBoost classifier with class balance handling.
    scale_pos_weight: ratio of negative/positive class to balance predictions
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,  # NEW: balances UP/DOWN bias
        random_state=42,
        n_jobs=-1
    )