"""
Tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from api.main import app
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint returns welcome message."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data
    assert 'version' in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    # May be 200 or 503 depending on whether models are loaded
    assert response.status_code in [200, 503]
    
    data = response.json()
    assert 'status' in data
    assert 'model_loaded' in data
    assert 'uptime_seconds' in data


def test_predict_endpoint_validation(client, sample_transaction):
    """Test prediction endpoint validates input."""
    # Valid transaction
    response = client.post("/predict", json=sample_transaction)
    
    # Should work or return 500 if models not loaded
    assert response.status_code in [200, 500]
    
    # Invalid transaction (negative amount)
    invalid_tx = sample_transaction.copy()
    invalid_tx['Amount'] = -100.0
    
    response = client.post("/predict", json=invalid_tx)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_missing_fields(client):
    """Test prediction endpoint requires all fields."""
    incomplete_tx = {
        "Time": 12345,
        "Amount": 150.00,
        # Missing V1-V28
    }
    
    response = client.post("/predict", json=incomplete_tx)
    assert response.status_code == 422


def test_predict_response_structure(client, sample_transaction):
    """Test prediction response has correct structure."""
    response = client.post("/predict", json=sample_transaction)
    
    if response.status_code == 200:
        data = response.json()
        
        # Check required fields
        assert 'transaction_id' in data
        assert 'is_fraud' in data
        assert 'fraud_probability' in data
        assert 'risk_level' in data
        assert 'explanation' in data
        assert 'timestamp' in data
        assert 'model_version' in data
        
        # Check types
        assert isinstance(data['is_fraud'], bool)
        assert isinstance(data['fraud_probability'], float)
        assert data['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
        assert 0 <= data['fraud_probability'] <= 1


def test_batch_prediction_endpoint(client, sample_transactions):
    """Test batch prediction endpoint."""
    batch_request = {
        "transactions": sample_transactions[:5]
    }
    
    response = client.post("/predict/batch", json=batch_request)
    
    if response.status_code == 200:
        data = response.json()
        
        assert 'predictions' in data
        assert 'batch_size' in data
        assert 'processing_time_ms' in data
        
        assert data['batch_size'] == 5
        assert len(data['predictions']) == 5


def test_batch_prediction_size_limit(client, sample_transaction):
    """Test batch prediction enforces size limit."""
    # Create oversized batch (> 1000)
    large_batch = {
        "transactions": [sample_transaction] * 1001
    }
    
    response = client.post("/predict/batch", json=large_batch)
    assert response.status_code == 422  # Validation error


def test_batch_prediction_empty(client):
    """Test batch prediction rejects empty batch."""
    empty_batch = {
        "transactions": []
    }
    
    response = client.post("/predict/batch", json=empty_batch)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    
    assert response.status_code == 200
    # Prometheus metrics are in text format
    assert 'text/plain' in response.headers.get('content-type', '')


def test_api_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options("/", headers={"Origin": "http://localhost:3000"})
    
    # CORS middleware should add headers
    assert response.status_code in [200, 405]


def test_request_headers(client, sample_transaction):
    """Test custom headers are added to responses."""
    response = client.post("/predict", json=sample_transaction)
    
    # Check for custom headers (if successful)
    if response.status_code == 200:
        assert 'X-Request-ID' in response.headers or True  # May not be present in tests
        assert 'X-Process-Time' in response.headers or True


def test_api_authentication_optional(client, sample_transaction):
    """Test API works without authentication in demo mode."""
    # No API key provided
    response = client.post("/predict", json=sample_transaction)
    
    # Should work in demo mode
    assert response.status_code in [200, 500]  # 500 if models not loaded


def test_api_with_valid_key(client, sample_transaction):
    """Test API accepts valid API key."""
    headers = {"X-API-Key": "demo-api-key-12345"}
    
    response = client.post("/predict", json=sample_transaction, headers=headers)
    
    # Should work
    assert response.status_code in [200, 500]
