from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class TransactionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Time: float = Field(..., description="Seconds elapsed between this transaction and the first transaction in the dataset", examples=[0.0])
    V1: float = Field(..., description="PCA component 1", examples=[-1.3598071336738])
    V2: float = Field(..., description="PCA component 2", examples=[-0.0727811733098497])
    V3: float = Field(..., description="PCA component 3", examples=[2.53634673796914])
    V4: float = Field(..., description="PCA component 4", examples=[1.37815522427443])
    V5: float = Field(..., description="PCA component 5", examples=[-0.338320769942518])
    V6: float = Field(..., description="PCA component 6", examples=[0.462387777762292])
    V7: float = Field(..., description="PCA component 7", examples=[0.239598554061257])
    V8: float = Field(..., description="PCA component 8", examples=[0.0986979012610507])
    V9: float = Field(..., description="PCA component 9", examples=[0.363786969611213])
    V10: float = Field(..., description="PCA component 10", examples=[0.0907941719789316])
    V11: float = Field(..., description="PCA component 11", examples=[-0.551599533260813])
    V12: float = Field(..., description="PCA component 12", examples=[-0.617800855762348])
    V13: float = Field(..., description="PCA component 13", examples=[-0.991389847235408])
    V14: float = Field(..., description="PCA component 14", examples=[-0.311169353699879])
    V15: float = Field(..., description="PCA component 15", examples=[1.46817697209427])
    V16: float = Field(..., description="PCA component 16", examples=[-0.470400525259478])
    V17: float = Field(..., description="PCA component 17", examples=[0.207971241929242])
    V18: float = Field(..., description="PCA component 18", examples=[0.0257905801985591])
    V19: float = Field(..., description="PCA component 19", examples=[0.403992960255733])
    V20: float = Field(..., description="PCA component 20", examples=[0.251412098239705])
    V21: float = Field(..., description="PCA component 21", examples=[-0.018306777944153])
    V22: float = Field(..., description="PCA component 22", examples=[0.277837575558899])
    V23: float = Field(..., description="PCA component 23", examples=[-0.110473910188767])
    V24: float = Field(..., description="PCA component 24", examples=[0.0669280749146731])
    V25: float = Field(..., description="PCA component 25", examples=[0.128539358273528])
    V26: float = Field(..., description="PCA component 26", examples=[-0.189114843888824])
    V27: float = Field(..., description="PCA component 27", examples=[0.133558376740387])
    V28: float = Field(..., description="PCA component 28", examples=[-0.0210530534538215])
    Amount: float = Field(..., description="Transaction Amount", examples=[149.62])

class BatchTransactionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    transactions: List[TransactionRequest] = Field(..., description="List of transactions for prediction")

class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    transaction_id: str = Field(..., description="Unique identifier for the transaction request")
    fraud_probability: float = Field(..., description="Probability [0, 1] that the transaction is fraudulent")
    is_fraud: bool = Field(..., description="Boolean indicating if transaction is considered fraud")
    risk_level: RiskLevel = Field(..., description="Risk level classification based on probability")
    processing_time_ms: float = Field(..., description="Time taken to process the prediction in milliseconds")

class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    predictions: List[PredictionResponse] = Field(..., description="List of prediction responses")
    total_processing_time_ms: float = Field(..., description="Total time taken for batch prediction in milliseconds")

class HealthResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    status: str = Field(..., description="Status of the API")
    model_loaded: bool = Field(..., description="Boolean indicating if the ML model is loaded successfully")
    version: str = Field(..., description="API Version")

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model_type: str = Field(..., description="Type of the loaded model")
    training_date: str = Field(..., description="Date when the model was trained")
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics of the model")
    threshold: float = Field(..., description="Probability threshold used for classification")
