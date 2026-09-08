"""
Unit tests for model inference and fraud risk classification.

Tests prediction range boundaries, response dictionary schemas,
risk level classification thresholds, and batch size consistency using mocked models.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.models.predict import FraudDetector


class DummyEstimator:
    """Mock ML estimator simulating XGBoost / scikit-learn classifier interface."""

    def __init__(self, probabilities: np.ndarray = None):
        self.probabilities = probabilities

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        if self.probabilities is not None:
            # If specific probabilities provided, ensure length matches
            if len(self.probabilities) == n_samples:
                probs = self.probabilities
            else:
                probs = np.resize(self.probabilities, n_samples)
        else:
            probs = np.random.uniform(0.01, 0.99, size=n_samples)

        # Return 2D array [prob_class_0, prob_class_1]
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


@pytest.fixture
def mock_detector() -> FraudDetector:
    """
    Instantiate a FraudDetector initialized with a Mock model without disk dependency.
    """
    detector = FraudDetector(auto_load=False)
    detector.xgb_model = DummyEstimator()
    detector.model = detector.xgb_model
    detector.threshold = 0.5
    detector.config = {
        "thresholds": {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.85,
            "critical": 0.95,
        }
    }
    return detector


@pytest.fixture
def sample_transaction() -> Dict[str, Any]:
    """
    Generate a single realistic transaction payload.
    """
    sample = {
        "Time": 43200.0,
        "Amount": 149.62,
    }
    for i in range(1, 29):
        sample[f"V{i}"] = 0.05 * i - 0.7
    return sample


@pytest.fixture
def sample_batch_transactions(sample_transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a list of transaction dictionaries for batch inference testing.
    """
    batch = []
    for i in range(10):
        tx = sample_transaction.copy()
        tx["Time"] = sample_transaction["Time"] + i * 100
        tx["Amount"] = sample_transaction["Amount"] + i * 25.5
        batch.append(tx)
    return batch


def test_prediction_in_range(mock_detector: FraudDetector, sample_transaction: Dict[str, Any]):
    """
    Verify that single and batch fraud probabilities are strictly bounded within [0.0, 1.0].
    """
    # Test single prediction
    result = mock_detector.predict_single(sample_transaction)
    assert 0.0 <= result["fraud_probability"] <= 1.0, "Probability must be in range [0, 1]"

    # Test with forced extreme bounds
    mock_detector.xgb_model = DummyEstimator(probabilities=np.array([0.0, 0.5, 1.0]))
    extreme_batch = [sample_transaction.copy() for _ in range(3)]
    batch_results = mock_detector.predict_batch(extreme_batch)

    for res in batch_results:
        assert 0.0 <= res["fraud_probability"] <= 1.0
        assert isinstance(res["fraud_probability"], float)


def test_prediction_output_format(mock_detector: FraudDetector, sample_transaction: Dict[str, Any]):
    """
    Verify that single prediction returns a dictionary containing all required keys and valid types:
    - 'fraud_probability' (float)
    - 'is_fraud' (bool)
    - 'risk_level' (str)
    """
    result = mock_detector.predict_single(sample_transaction)

    assert isinstance(result, dict), "Prediction output must be a dictionary"
    assert "fraud_probability" in result, "Missing 'fraud_probability' key"
    assert "is_fraud" in result, "Missing 'is_fraud' key"
    assert "risk_level" in result, "Missing 'risk_level' key"

    assert isinstance(result["fraud_probability"], float), "'fraud_probability' must be a float"
    assert isinstance(result["is_fraud"], bool), "'is_fraud' must be a boolean"
    assert isinstance(result["risk_level"], str), "'risk_level' must be a string"
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}, f"Invalid risk level: {result['risk_level']}"


def test_risk_level_classification(mock_detector: FraudDetector):
    """
    Verify risk level mapping conforms to system threshold specifications:
    - prob < 0.30 -> 'LOW'
    - 0.30 <= prob < 0.60 -> 'MEDIUM'
    - 0.60 <= prob < 0.85 -> 'HIGH'
    - prob >= 0.85 -> 'CRITICAL'
    """
    # Test boundary conditions
    assert mock_detector._get_risk_level(0.0) == "LOW"
    assert mock_detector._get_risk_level(0.29) == "LOW"
    assert mock_detector._get_risk_level(0.30) == "MEDIUM"
    assert mock_detector._get_risk_level(0.59) == "MEDIUM"
    assert mock_detector._get_risk_level(0.60) == "HIGH"
    assert mock_detector._get_risk_level(0.84) == "HIGH"
    assert mock_detector._get_risk_level(0.85) == "CRITICAL"
    assert mock_detector._get_risk_level(0.99) == "CRITICAL"
    assert mock_detector._get_risk_level(1.0) == "CRITICAL"


def test_batch_prediction_length(
    mock_detector: FraudDetector, sample_batch_transactions: List[Dict[str, Any]]
):
    """
    Verify that batch prediction output length precisely matches input transaction length.
    """
    n_expected = len(sample_batch_transactions)
    results = mock_detector.predict_batch(sample_batch_transactions)

    assert isinstance(results, list), "Batch prediction output must be a list"
    assert len(results) == n_expected, f"Expected {n_expected} predictions, got {len(results)}"

    # Check that each prediction item in the batch conforms to schema
    for res in results:
        assert "fraud_probability" in res
        assert "is_fraud" in res
        assert "risk_level" in res
        assert 0.0 <= res["fraud_probability"] <= 1.0
