import os
import yaml
import joblib
import logging
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.features.engineering import feature_engineering_pipeline

logger = get_logger(__name__)


class FraudDetector:
    """Class to perform fraud prediction on transactions."""

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        config_path: str = "configs/config.yaml",
        model_dir: Optional[str] = None,
        auto_load: bool = True,
    ):
        """Initialize detector with loaded model artifacts and config."""
        self.artifacts_dir = model_dir or artifacts_dir
        self.config_path = config_path
        self.xgb_model = None
        self.iso_model = None
        self.model = None
        self.threshold = 0.5
        self.config = {}

        if auto_load:
            try:
                self.load_model()
            except Exception as e:
                logger.warning(
                    f"Could not auto-load artifacts from '{self.artifacts_dir}': {e}"
                )

    def load_model(self) -> None:
        """Load model artifacts and metadata from disk."""
        logger.info(f"Loading FraudDetector artifacts from {self.artifacts_dir}...")
        xgb_path = os.path.join(self.artifacts_dir, "xgb_model.joblib")
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            self.model = self.xgb_model
        else:
            raise FileNotFoundError(f"Model file not found: {xgb_path}")

        iso_path = os.path.join(self.artifacts_dir, "iso_model.joblib")
        if os.path.exists(iso_path):
            self.iso_model = joblib.load(iso_path)

        meta_path = os.path.join(self.artifacts_dir, "metadata.yaml")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = yaml.safe_load(f) or {}
            self.threshold = metadata.get("optimal_threshold", 0.5)

        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}

        logger.info("FraudDetector initialized successfully.")

    def _get_risk_level(self, probability: float) -> str:
        """Get risk level based on fraud probability."""
        if probability < 0.3:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.85:
            return "HIGH"
        else:
            return "CRITICAL"

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and preprocessing to raw input."""
        processed = df.copy()
        
        # Apply feature engineering (creates time/amount/interaction features)
        processed = feature_engineering_pipeline(processed, self.config)
        
        # Scale Amount -> scaled_amount (using simple normalization for inference)
        if "Amount" in processed.columns:
            processed["scaled_amount"] = processed["Amount"]
            processed = processed.drop(columns=["Amount"])
        
        # Normalize Time -> scaled_time (seconds to hours)
        if "Time" in processed.columns:
            processed["scaled_time"] = processed["Time"] / 3600.0
            processed = processed.drop(columns=["Time"])
        
        # Drop non-numeric columns (e.g., amount_bin is categorical)
        non_numeric = processed.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            processed = processed.drop(columns=non_numeric)
        
        # Drop target column if accidentally present
        for col in ["Class", "is_fraud"]:
            if col in processed.columns:
                processed = processed.drop(columns=[col])
        
        return processed

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability array for a DataFrame."""
        active_model = self.xgb_model or self.model
        if active_model is None:
            raise RuntimeError("No model is loaded for prediction.")

        processed_df = self._preprocess(df)
        if hasattr(active_model, "predict_proba"):
            probs = active_model.predict_proba(processed_df)
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs.flatten()
        elif hasattr(active_model, "predict"):
            preds = active_model.predict(processed_df)
            return np.asarray(preds, dtype=float)
        raise AttributeError("Loaded model does not support prediction.")

    def predict_single(
        self, transaction: Dict[str, Any]
    ) -> Dict[str, Union[float, bool, str]]:
        """Predict fraud probability for a single transaction."""
        logger.info(
            f"Predicting single transaction: {transaction.get('transaction_id', 'unknown')}"
        )
        try:
            df = pd.DataFrame([transaction])
            prob = float(self.predict_proba(df)[0])
            is_fraud = bool(prob >= self.threshold)
            risk_level = self._get_risk_level(prob)

            result = {
                "fraud_probability": prob,
                "is_fraud": is_fraud,
                "risk_level": risk_level,
            }
            logger.info(f"Prediction result: {result}")
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Union[float, bool, str]]]:
        """Predict fraud probability for a batch of transactions."""
        logger.info(f"Predicting batch of {len(transactions)} transactions...")
        try:
            df = pd.DataFrame(transactions)
            probs = self.predict_proba(df)

            results = []
            for prob in probs:
                prob_float = float(prob)
                results.append(
                    {
                        "fraud_probability": prob_float,
                        "is_fraud": bool(prob_float >= self.threshold),
                        "risk_level": self._get_risk_level(prob_float),
                    }
                )

            logger.info("Batch prediction completed successfully.")
            return results
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

