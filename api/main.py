"""
FastAPI application for fraud detection service.

Provides REST API for real-time fraud detection with <100ms latency.
"""

import os
import time
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
from starlette.responses import Response

from api.schemas import (
    TransactionInput, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthResponse, ErrorResponse
)
from api.inference import get_service, FraudDetectionService
from api.middleware import rate_limit_middleware, api_key_middleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application startup time
START_TIME = time.time()

# Prometheus metrics
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    'fraud_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    'fraud_api_request_duration_seconds',
    'API request latency',
    ['method', 'endpoint'],
    registry=registry
)

PREDICTION_COUNTER = Counter(
    'fraud_predictions_total',
    'Total predictions made',
    ['prediction'],
    registry=registry
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("="*70)
    logger.info("FRAUD DETECTION API - Starting up...")
    logger.info("="*70)
    
    # Load models at startup
    try:
        service = get_service()
        is_healthy, status = service.is_healthy()
        
        if is_healthy:
            logger.info("✓ Service initialized successfully")
            logger.info(f"  - Model loaded: {status['model_loaded']}")
            logger.info(f"  - Scaler loaded: {status['scaler_loaded']}")
            logger.info(f"  - Redis connected: {status['redis_connected']}")
        else:
            logger.error("✗ Service initialization failed")
            for key, value in status.items():
                logger.error(f"  - {key}: {value}")
    except Exception as e:
        logger.error(f"✗ Startup failed: {str(e)}")
        raise
    
    logger.info("="*70)
    logger.info("API ready at http://localhost:8000")
    logger.info("Docs available at http://localhost:8000/docs")
    logger.info("="*70)
    
    yield
    
    # Shutdown
    logger.info("Shutting down fraud detection API...")


# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud detection with ML ensemble",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allow all for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with latency."""
    start_time = time.time()
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Process request
    response = await call_next(request)
    
    # Calculate latency
    latency = (time.time() - start_time) * 1000
    
    # Log request
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Latency: {latency:.2f}ms"
    )
    
    # Update Prometheus metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(latency / 1000)  # Convert to seconds
    
    # Add latency header
    response.headers['X-Process-Time'] = str(latency)
    response.headers['X-Request-ID'] = request_id
    
    return response


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status, model info, and connection status.
    """
    service = get_service()
    is_healthy, status = service.is_healthy()
    
    uptime = time.time() - START_TIME
    
    # Check database (placeholder - not implemented in this version)
    database_connected = True  # Would actually check PostgreSQL connection
    
    health_response = HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=status['model_loaded'],
        model_version="fraud_ensemble_v1",
        uptime_seconds=uptime,
        redis_connected=status['redis_connected'],
        database_connected=database_connected,
        timestamp=datetime.now()
    )
    
    if not is_healthy:
        return JSONResponse(
            status_code=503,
            content=health_response.dict()
        )
    
    return health_response


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_fraud(
    transaction: TransactionInput,
    request: Request
):
    """
    Predict fraud for a single transaction.
    
    Returns fraud probability, binary prediction, risk level, and explanation.
    Target latency: <100ms
    """
    try:
        service = get_service()
        
        # Make prediction
        result = service.predict_single(transaction.dict())
        
        # Update prediction counter
        PREDICTION_COUNTER.labels(
            prediction='fraud' if result['is_fraud'] else 'legitimate'
        ).inc()
        
        # Build response
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        response = PredictionResponse(
            transaction_id=transaction_id,
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            explanation=result['explanation'],
            timestamp=datetime.now(),
            model_version="fraud_ensemble_v1"
        )
        
        # Log high-risk predictions
        if result['risk_level'] in ['MEDIUM', 'HIGH']:
            logger.warning(
                f"HIGH RISK TRANSACTION: {transaction_id} - "
                f"Probability: {result['fraud_probability']:.4f}"
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    batch: BatchPredictionRequest,
    request: Request
):
    """
    Predict fraud for multiple transactions in batch.
    
    More efficient than individual predictions for large volumes.
    Maximum batch size: 1000 transactions.
    """
    start_time = time.time()
    
    try:
        service = get_service()
        
        # Convert to list of dicts
        transactions = [tx.dict() for tx in batch.transactions]
        
        # Make batch prediction
        results = service.predict_batch(transactions)
        
        # Build responses
        predictions = []
        for i, result in enumerate(results):
            transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:04d}"
            
            pred = PredictionResponse(
                transaction_id=transaction_id,
                is_fraud=result['is_fraud'],
                fraud_probability=result['fraud_probability'],
                risk_level=result['risk_level'],
                explanation=result['explanation'],
                timestamp=datetime.now(),
                model_version="fraud_ensemble_v1"
            )
            predictions.append(pred)
            
            # Update counter
            PREDICTION_COUNTER.labels(
                prediction='fraud' if result['is_fraud'] else 'legitimate'
            ).inc()
        
        processing_time = (time.time() - start_time) * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Batch prediction: {len(predictions)} transactions in {processing_time:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes request counts, latencies, and prediction distributions.
    """
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    error_response = ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        timestamp=datetime.now()
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('API_PORT', 8000))
    host = os.getenv('API_HOST', '0.0.0.0')
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
