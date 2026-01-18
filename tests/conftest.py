"""
Pytest configuration and fixtures for fraud detection tests.
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture
def sample_transaction():
    """Generate a sample transaction for testing."""
    transaction = {
        "Time": 12345,
        "Amount": 150.00,
    }
    
    # Add V1-V28 features
    for i in range(1, 29):
        transaction[f"V{i}"] = np.random.randn()
    
    return transaction


@pytest.fixture
def sample_transactions():
    """Generate multiple sample transactions."""
    transactions = []
    for _ in range(10):
        tx = {
            "Time": np.random.randint(0, 172800),
            "Amount": np.random.uniform(1, 1000),
        }
        for i in range(1, 29):
            tx[f"V{i}"] = np.random.randn()
        transactions.append(tx)
    
    return transactions


@pytest.fixture
def sample_dataframe():
    """Generate a sample DataFrame for testing."""
    n_samples = 100
    data = {
        "Time": np.random.randint(0, 172800, n_samples),
        "Amount": np.random.lognormal(3, 2, n_samples),
    }
    
    # Add V1-V28
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_samples)
    
    # Add target
    data["Class"] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    
    return pd.DataFrame(data)


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock model for testing without loading actual models."""
    class MockModel:
        def predict(self, X):
            return np.zeros(len(X))
        
        def predict_proba(self, X):
            proba = np.random.rand(len(X), 2)
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba
    
    return MockModel()


@pytest.fixture
def test_data_path(tmp_path):
    """Create temporary test data file."""
    df = pd.DataFrame({
        "Time": [1, 2, 3],
        "Amount": [10.0, 20.0, 30.0],
        **{f"V{i}": [0.0, 0.0, 0.0] for i in range(1, 29)},
        "Class": [0, 1, 0]
    })
    
    filepath = tmp_path / "test_data.csv"
    df.to_csv(filepath, index=False)
    
    return str(filepath)
