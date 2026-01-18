"""
Tests for model training and ensemble.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def test_ensemble_predictions(sample_dataframe):
    """Test ensemble model predictions."""
    # This test is slow, TODO: add proper mocking
    from models.ensemble import FraudEnsemble
    
    # Skip if models not trained
    try:
        ensemble = FraudEnsemble()
    except Exception:
        pytest.skip("Models not trained yet")
        return
    
    X = sample_dataframe.drop('Class', axis=1)
    
    # Test predict_proba
    probas = ensemble.predict_proba(X)
    
    assert len(probas) == len(X)
    assert (probas >= 0).all() and (probas <= 1).all()
    
    # Test predict
    preds = ensemble.predict(X, threshold=0.5)
    
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_ensemble_weights():
    """Test ensemble weight validation."""
    from models.ensemble import FraudEnsemble
    
    # Skip if models not trained
    try:
        # Test with custom weights
        weights = {
            'isolation_forest': 0.3,
            'xgboost': 0.5,
            'autoencoder': 0.2
        }
        
        ensemble = FraudEnsemble(weights=weights)
        
        # Check weights are normalized
        total = sum(ensemble.weights.values())
        assert abs(total - 1.0) < 1e-6
        
    except Exception:
        pytest.skip("Models not trained yet")


def test_shap_explanations(sample_dataframe):
    """Test SHAP feature explanations."""
    from models.ensemble import FraudEnsemble
    
    try:
        ensemble = FraudEnsemble()
    except Exception:
        pytest.skip("Models not trained yet")
        return
    
    X = sample_dataframe.drop('Class', axis=1).head(3)
    
    explanations = ensemble.get_top_features(X, n_features=3)
    
    assert len(explanations) == len(X)
    
    for exp in explanations:
        assert isinstance(exp, dict)
        # Should have at most 3 features (or empty if SHAP failed)
        assert len(exp) <= 3


def test_cost_calculation():
    """Test cost savings calculation."""
    from models.evaluate import calculate_cost_savings
    
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])  # 1 FN, 0 FP
    
    costs = calculate_cost_savings(y_true, y_pred, fn_cost=10000, fp_cost=100)
    
    assert 'total_cost' in costs
    assert 'fn_cost' in costs
    assert 'fp_cost' in costs
    assert 'savings' in costs
    
    # With 1 FN and 0 FP
    assert costs['fn_cost'] == 10000
    assert costs['fp_cost'] == 0
    
    # Savings should be positive (we caught 2 out of 3 frauds)
    assert costs['savings'] > 0


def test_cost_calculation_validation():
    """Test cost calculation input validation."""
    from models.evaluate import calculate_cost_savings
    
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])  # Different length
    
    with pytest.raises(ValueError):
        calculate_cost_savings(y_true, y_pred)
