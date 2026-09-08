import time
import uuid
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd

from src.api.schemas import (
    TransactionRequest,
    BatchTransactionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    RiskLevel
)
from src.models.predict import FraudDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fraud_api")

# Global reference for our model
model_instance: FraudDetector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to load the model on startup and clean up on shutdown."""
    global model_instance
    logger.info("Starting up API and loading model...")
    try:
        model_instance = FraudDetector(model_dir="artifacts/")
        model_instance.load_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Not failing the startup here so health check can report model_loaded=False
    yield
    logger.info("Shutting down API...")
    # Clean up resources if needed
    model_instance = None

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions in real-time.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def determine_risk_level(probability: float) -> RiskLevel:
    """Determine risk level based on fraud probability."""
    if probability < 0.2:
        return RiskLevel.LOW
    elif probability < 0.5:
        return RiskLevel.MEDIUM
    elif probability < 0.8:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_instance is not None and getattr(model_instance, 'model', None) is not None,
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if not model_instance or not getattr(model_instance, 'model', None):
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    # Normally, these would be loaded from a metadata file in artifacts/
    return ModelInfoResponse(
        model_type="RandomForestClassifier",
        training_date="2026-08-29",
        metrics={"f1_score": 0.85, "precision": 0.9, "recall": 0.8},
        threshold=0.5
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    """Predict if a single transaction is fraudulent."""
    if not model_instance or not getattr(model_instance, 'model', None):
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot process request.")
    
    start_time = time.perf_counter()
    transaction_id = str(uuid.uuid4())
    logger.info(f"Processing prediction for transaction {transaction_id}")
    
    try:
        result = model_instance.predict_single(transaction.model_dump())
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=result["fraud_probability"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            processing_time_ms=processing_time_ms
        )
    except Exception as e:
        logger.error(f"Error processing transaction {transaction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchTransactionRequest):
    """Predict fraud for a batch of transactions."""
    if not model_instance or not getattr(model_instance, 'model', None):
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    start_time = time.perf_counter()
    logger.info(f"Processing batch prediction for {len(request.transactions)} transactions")
    
    try:
        transactions_data = [t.model_dump() for t in request.transactions]
        results = model_instance.predict_batch(transactions_data)
        
        predictions = []
        for result in results:
            predictions.append(
                PredictionResponse(
                    transaction_id=str(uuid.uuid4()),
                    fraud_probability=result["fraud_probability"],
                    is_fraud=result["is_fraud"],
                    risk_level=result["risk_level"],
                    processing_time_ms=0.0
                )
            )
            
        total_time_ms = (time.perf_counter() - start_time) * 1000
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time_ms
        )
    except Exception as e:
        logger.error(f"Error processing batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
