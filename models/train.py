# models/train.py
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)
from loguru import logger
from models.xgb_model import build_xgb_model


def train_xgb(X, y):
    """
    Professional 4H training pipeline with:
    - TimeSeriesSplit cross-validation
    - Class imbalance handling (scale_pos_weight)
    - Multiple evaluation metrics
    - Feature importance tracking
    """
    logger.info(f"Training XGBoost on {len(X)} samples, {X.shape[1]} features")

    # Calculate class balance ratio for XGBoost
    n_down = int((y == 0).sum())
    n_up   = int((y == 1).sum())
    scale_pos_weight = n_down / n_up if n_up > 0 else 1.0
    logger.info(f"Class balance — UP: {n_up} | DOWN: {n_down} | scale_pos_weight: {scale_pos_weight:.3f}")

    # -----------------------------------------------
    # TIME SERIES CROSS VALIDATION
    # -----------------------------------------------
    tscv = TimeSeriesSplit(n_splits=5)
    cv_accuracies = []
    cv_f1_scores  = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = build_xgb_model(scale_pos_weight=scale_pos_weight)
        fold_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = fold_model.predict(X_val)
        acc   = accuracy_score(y_val, preds)
        f1    = f1_score(y_val, preds, average="weighted", zero_division=0)
        cv_accuracies.append(acc)
        cv_f1_scores.append(f1)
        logger.info(f"Fold {fold+1}: Accuracy={acc:.4f} | F1={f1:.4f}")

    logger.info(f"CV Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
    logger.info(f"CV F1 Score: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")

    # -----------------------------------------------
    # FINAL MODEL
    # -----------------------------------------------
    split   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = build_xgb_model(scale_pos_weight=scale_pos_weight)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":         accuracy_score(y_test, preds),
        "f1":               f1_score(y_test, preds, average="weighted", zero_division=0),
        "precision":        precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall":           recall_score(y_test, preds, average="weighted", zero_division=0),
        "roc_auc":          roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else 0.0,
        "cv_accuracy_mean": float(np.mean(cv_accuracies)),
        "cv_accuracy_std":  float(np.std(cv_accuracies)),
        "cv_f1_mean":       float(np.mean(cv_f1_scores)),
    }

    logger.info("=" * 50)
    logger.info("FINAL MODEL EVALUATION")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info("=" * 50)
    logger.info("\n" + classification_report(y_test, preds, zero_division=0))

    feat_importance = dict(zip(X.columns, model.feature_importances_))
    top_features    = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Top 5 Features:")
    for feat, imp in top_features:
        logger.info(f"  {feat}: {imp:.4f}")

    return model, metrics