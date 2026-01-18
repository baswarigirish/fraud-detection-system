"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class TransactionInput(BaseModel):
    """Input schema for a single transaction."""
    
    Time: int = Field(..., description="Seconds elapsed since first transaction")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., description="Transaction amount", ge=0.0)
    
    @validator('Amount')
    def amount_must_be_positive(cls, v):
        """Validate that amount is non-negative."""
        if v < 0:
            raise ValueError('Amount must be positive or zero')
        return v
    
    @validator('Time')
    def time_must_be_valid(cls, v):
        """Validate that time is non-negative."""
        if v < 0:
            raise ValueError('Time must be non-negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 12345,
                "V1": -0.5, "V2": 0.3, "V3": 1.2, "V4": -0.8,
                "V5": 0.1, "V6": -0.3, "V7": 0.5, "V8": -0.2,
                "V9": 0.7, "V10": -0.4, "V11": 0.2, "V12": 0.9,
                "V13": -0.6, "V14": 0.4, "V15": -0.1, "V16": 0.8,
                "V17": -0.3, "V18": 0.6, "V19": -0.5, "V20": 0.2,
                "V21": -0.7, "V22": 0.4, "V23": -0.2, "V24": 0.5,
                "V25": 0.3, "V26": -0.4, "V27": 0.1, "V28": -0.6,
                "Amount": 150.00
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for fraud prediction."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    fraud_probability: float = Field(..., description="Fraud probability (0-1)", ge=0.0, le=1.0)
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    explanation: Dict[str, float] = Field(..., description="Top contributing features (SHAP values)")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_20250113_123456_abc",
                "is_fraud": False,
                "fraud_probability": 0.23,
                "risk_level": "LOW",
                "explanation": {
                    "V14": 0.15,
                    "V4": -0.08,
                    "Amount_Log": 0.05
                },
                "timestamp": "2025-01-13T10:30:00",
                "model_version": "fraud_ensemble_v1"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    transactions: List[TransactionInput] = Field(
        ...,
        description="List of transactions to predict",
        max_items=1000
    )
    
    @validator('transactions')
    def check_batch_size(cls, v):
        """Validate batch size."""
        if len(v) == 0:
            raise ValueError('Batch cannot be empty')
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse]
    batch_size: int
    processing_time_ms: float
    timestamp: datetime


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status: healthy or unhealthy")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    model_version: str = Field(..., description="Current model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    redis_connected: bool = Field(..., description="Redis connection status")
    database_connected: bool = Field(..., description="PostgreSQL connection status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "fraud_ensemble_v1",
                "uptime_seconds": 3600.5,
                "redis_connected": True,
                "database_connected": True,
                "timestamp": "2025-01-13T10:30:00"
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")
