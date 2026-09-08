import os
import yaml
import joblib
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, f1_score

from src.utils.logger import get_logger
from src.features.engineering import feature_engineering_pipeline

logger = get_logger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load project configuration."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, config: Dict[str, Any]) -> XGBClassifier:
    """Train XGBoost model with early stopping."""
    logger.info("Training XGBoost model...")
    params = dict(config.get("models", {}).get("xgboost", {}))
    
    # Calculate scale_pos_weight from actual data distribution
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    params["scale_pos_weight"] = scale_pos_weight
    
    # Extract params that need special handling
    early_stopping = params.pop("early_stopping_rounds", 50)
    eval_metric = params.pop("eval_metric", "aucpr")
    
    # In XGBoost 2.0+, early_stopping_rounds is a constructor param
    model = XGBClassifier(
        **params,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    logger.info("XGBoost training completed.")
    return model

def train_isolation_forest(X_train: pd.DataFrame, config: Dict[str, Any]) -> IsolationForest:
    """Train Isolation Forest for anomaly detection."""
    logger.info("Training Isolation Forest model...")
    params = config.get("models", {}).get("isolation_forest", {})
    model = IsolationForest(**params)
    model.fit(X_train)
    logger.info("Isolation Forest training completed.")
    return model

def find_optimal_threshold(model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Find threshold that maximizes F1 score using precision-recall curve."""
    logger.info("Finding optimal threshold...")
    y_prob = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    
    # Calculate F1 scores handling zero division
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    logger.info(f"Optimal threshold found: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
    return float(optimal_threshold)

def create_ensemble(xgb_model: XGBClassifier, iso_model: IsolationForest, weights: Dict[str, float], threshold: float) -> Dict[str, Any]:
    """Create ensemble artifact dictionary."""
    logger.info("Creating model artifacts...")
    return {
        "xgb_model": xgb_model,
        "iso_model": iso_model,
        "weights": weights,
        "optimal_threshold": threshold
    }

def save_artifacts(artifacts: Dict[str, Any], output_dir: str) -> None:
    """Save model artifacts to disk."""
    logger.info(f"Saving artifacts to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, artifact in artifacts.items():
        if name not in ("optimal_threshold", "weights"):
            path = os.path.join(output_dir, f"{name}.joblib")
            joblib.dump(artifact, path)
            logger.info(f"Saved {name} to {path}")
            
    # Save metadata/threshold
    meta_path = os.path.join(output_dir, "metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.dump({
            "optimal_threshold": artifacts["optimal_threshold"],
            "weights": artifacts["weights"]
        }, f)
    logger.info(f"Saved metadata to {meta_path}")

def main():
    config = load_config()
    data_dir = config.get("data", {}).get("processed_path", "data/processed")
    artifacts_dir = config.get("data", {}).get("artifacts_path", "artifacts")
    
    try:
        # Load data
        logger.info("Loading processed data...")
        train_data = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        val_data = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
        
        target_col = config.get("data", {}).get("target_column", "Class")
        
        X_train, y_train = train_data.drop(columns=[target_col]), train_data[target_col]
        X_val, y_val = val_data.drop(columns=[target_col]), val_data[target_col]
        
        # Feature engineering is already applied during preprocessing
        # Drop any non-numeric columns (e.g., amount_bin is categorical)
        non_numeric = X_train.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            logger.info(f"Dropping non-numeric columns: {non_numeric}")
            X_train = X_train.drop(columns=non_numeric)
            X_val = X_val.drop(columns=non_numeric)
        
        # Train models
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, config)
        iso_model = train_isolation_forest(X_train, config)
        
        # Find threshold
        threshold = find_optimal_threshold(xgb_model, X_val, y_val)
        
        # Ensemble & Save
        weights = {"xgb": 0.8, "iso": 0.2}  # default weights
        artifacts = create_ensemble(xgb_model, iso_model, weights, threshold)
        save_artifacts(artifacts, artifacts_dir)
        
        logger.info("Training pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
