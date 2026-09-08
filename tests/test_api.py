"""
Integration and functional tests for the FastAPI Fraud Detection API.

Tests health check endpoint, single prediction validation, invalid schema rejection (422),
and batch prediction throughput using FastAPI TestClient and mocked model backends.
"""

from typing import Any, Dict
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.api.main import app
from src.models.predict import FraudDetector


class MockEstimator:
    """Mock ML classifier for fast, deterministic API unit testing."""

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        probs = np.full((n, 2), [0.85, 0.15])  # Low probability default
        return probs

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


@pytest.fixture
def mock_fraud_detector() -> FraudDetector:
    """Provide an initialized mock FraudDetector instance."""
    detector = FraudDetector(auto_load=False)
    estimator = MockEstimator()
    detector.xgb_model = estimator
    detector.model = estimator
    detector.threshold = 0.5
    detector.config = {}
    return detector


@pytest.fixture
def client(mock_fraud_detector: FraudDetector) -> TestClient:
    """
    Provide FastAPI TestClient with injected mock model backend.
    """
    # Inject mocked detector directly into API module scope
    api_module.model_instance = mock_fraud_detector

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_transaction_payload() -> Dict[str, Any]:
    """
    Generate a valid payload dictionary satisfying the TransactionRequest schema.
    """
    payload = {
        "Time": 1000.0,
        "Amount": 149.62,
    }
    for i in range(1, 29):
        payload[f"V{i}"] = 0.01 * i - 0.15
    return payload


def test_health_endpoint(client: TestClient):
    """
    Test GET /health:
    - Returns HTTP 200 OK
    - Returns JSON payload with status='healthy'
    - Confirms 'model_loaded' boolean flag and version string
    """
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)
    assert data["version"] == "1.0.0"


def test_predict_valid_input(client: TestClient, valid_transaction_payload: Dict[str, Any]):
    """
    Test POST /predict with valid transaction payload:
    - Returns HTTP 200 OK
    - Validates presence of transaction_id, fraud_probability, is_fraud, risk_level, processing_time_ms
    - Validates mathematical bounds [0.0, 1.0] for probability
    """
    response = client.post("/predict", json=valid_transaction_payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_level" in data
    assert "processing_time_ms" in data

    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert isinstance(data["is_fraud"], bool)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert data["processing_time_ms"] >= 0.0


def test_predict_invalid_input(client: TestClient):
    """
    Test POST /predict with missing required fields:
    - Returns HTTP 422 Unprocessable Entity
    - Confirms request rejection on invalid data format
    """
    invalid_payload = {
        "Time": 100.0,
        "Amount": 50.0,
        # Missing all V1-V28 PCA feature columns
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422, f"Expected 422 validation error, got {response.status_code}"

    error_data = response.json()
    assert "detail" in error_data


def test_batch_predict(client: TestClient, valid_transaction_payload: Dict[str, Any]):
    """
    Test POST /predict/batch with multiple transactions:
    - Returns HTTP 200 OK
    - Verifies number of predictions matches number of submitted transactions
    - Verifies presence of total_processing_time_ms
    """
    batch_size = 5
    transactions = [valid_transaction_payload.copy() for _ in range(batch_size)]
    batch_payload = {"transactions": transactions}

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == batch_size
    assert "total_processing_time_ms" in data
    assert data["total_processing_time_ms"] >= 0.0

    for pred in data["predictions"]:
        assert "transaction_id" in pred
        assert "fraud_probability" in pred
        assert "is_fraud" in pred
        assert "risk_level" in pred
        assert 0.0 <= pred["fraud_probability"] <= 1.0
