# models/retrain.py
import os
import joblib
from datetime import datetime
from loguru import logger
from models.train import train_xgb


def retrain_model(X, y, save_path="models/model.pkl"):
    """
    Retrain model with versioning and performance tracking.
    Keeps backup of previous model before overwriting.
    """

    logger.info(f"Starting retraining pipeline — {datetime.utcnow().isoformat()}")

    # -----------------------------------------------
    # BACKUP previous model before overwriting
    # -----------------------------------------------
    if os.path.exists(save_path):
        backup_path = save_path.replace(".pkl", f"_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl")
        os.rename(save_path, backup_path)
        logger.info(f"Previous model backed up to {backup_path}")

    # -----------------------------------------------
    # TRAIN
    # -----------------------------------------------
    model, metrics = train_xgb(X, y)

    # -----------------------------------------------
    # SAVE with full metadata
    # -----------------------------------------------
    bundle = {
        "model": model,
        "metadata": {
            "trained_at":       datetime.utcnow().isoformat(),
            "accuracy":         metrics["accuracy"],
            "f1":               metrics["f1"],
            "precision":        metrics["precision"],
            "recall":           metrics["recall"],
            "roc_auc":          metrics["roc_auc"],
            "cv_accuracy_mean": metrics["cv_accuracy_mean"],
            "cv_accuracy_std":  metrics["cv_accuracy_std"],
            "cv_f1_mean":       metrics["cv_f1_mean"],
            "n_features":       len(model.feature_names_in_),
            "feature_names":    list(model.feature_names_in_),
        }
    }

    joblib.dump(bundle, save_path)
    logger.info(f"Model saved to {save_path}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics