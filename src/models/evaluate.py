import os
import yaml
import joblib
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, roc_auc_score
)
from src.utils.logger import get_logger
from src.features.engineering import feature_engineering_pipeline

logger = get_logger(__name__)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    logger.info("Evaluating model...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "auprc": average_precision_score(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred)
    }
    
    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
        
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return metrics

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true: pd.Series, y_scores: np.ndarray, save_path: str):
    """Plot and save precision-recall curve."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'AUPRC = {auprc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true: pd.Series, y_scores: np.ndarray, save_path: str):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names: List[str], save_path: str, top_n: int = 20):
    """Plot and save feature importances."""
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_. Skipping plot.")
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_report(metrics: Dict[str, float], plots_dir: str, output_path: str):
    """Generate evaluation markdown report."""
    report = "# Model Evaluation Report\n\n## Metrics\n"
    for k, v in metrics.items():
        report += f"- **{k.upper()}**: {v:.4f}\n"
        
    report += "\n## Plots\n"
    report += "Check the following plots in the plots directory:\n"
    report += "- Confusion Matrix\n- Precision-Recall Curve\n- ROC Curve\n- Feature Importance\n"
    
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Report generated at {output_path}")

def main():
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        data_dir = config.get("data", {}).get("processed_path", "data/processed")
        artifacts_dir = config.get("data", {}).get("artifacts_path", "artifacts")
        reports_dir = "reports"
        plots_dir = os.path.join(reports_dir, "figures")
        
        os.makedirs(plots_dir, exist_ok=True)
        
        logger.info("Loading test data...")
        test_data = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
        target_col = config.get("data", {}).get("target_column", "Class")
        
        X_test, y_test = test_data.drop(columns=[target_col]), test_data[target_col]
        
        # Drop non-numeric columns (feature engineering already done in preprocessing)
        non_numeric = X_test.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            logger.info(f"Dropping non-numeric columns: {non_numeric}")
            X_test = X_test.drop(columns=non_numeric)
        
        logger.info("Loading model artifacts...")
        xgb_model = joblib.load(os.path.join(artifacts_dir, "xgb_model.joblib"))
        
        with open(os.path.join(artifacts_dir, "metadata.yaml"), "r") as f:
            metadata = yaml.safe_load(f)
        threshold = metadata.get("optimal_threshold", 0.5)
        
        metrics = evaluate_model(xgb_model, X_test, y_test, threshold)
        
        # Recalculate prob and pred for plots
        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        logger.info("Generating plots...")
        plot_confusion_matrix(y_test, y_pred, os.path.join(plots_dir, "confusion_matrix.png"))
        plot_precision_recall_curve(y_test, y_prob, os.path.join(plots_dir, "pr_curve.png"))
        plot_roc_curve(y_test, y_prob, os.path.join(plots_dir, "roc_curve.png"))
        plot_feature_importance(xgb_model, X_test.columns.tolist(), os.path.join(plots_dir, "feature_importance.png"))
        
        generate_report(metrics, plots_dir, os.path.join(reports_dir, "evaluation_report.md"))
        
        logger.info("Evaluation completed successfully.")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
