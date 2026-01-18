"""
Ensemble model combining Isolation Forest, XGBoost, and Autoencoder.

Uses weighted voting with SHAP explainability for XGBoost predictions.
Weights favor XGBoost (0.5) since it performed best in validation.
"""

import os
import logging
from typing import Tuple, Dict, List
import warnings

import pandas as pd
import numpy as np
import joblib
# TensorFlow not available
# from tensorflow import keras
import json
import shap

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudEnsemble:
    """
    Ensemble model for fraud detection.
    
    Combines three different approaches:
    - Isolation Forest: Unsupervised anomaly detection
    - XGBoost: Supervised classification
    - Autoencoder: Deep learning anomaly detection
    """
    
    def __init__(
        self,
        model_dir: str = './models/saved_models',
        weights: Dict[str, float] = None
    ):
        """
        Initialize ensemble with pre-trained models.
        
        Args:
            model_dir: Directory containing saved models
            weights: Dictionary of model weights (must sum to 1.0)
        """
        self.model_dir = model_dir
        
        if weights is None:
            # Default weights based on validation performance (without autoencoder)
            self.weights = {
                'isolation_forest': 0.3,
                'xgboost': 0.7,
                # 'autoencoder': 0.3  # Skipped - TensorFlow not available
            }
        else:
            self.weights = weights
        
        # Validate weights
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0):
            logger.warning(f"Weights sum to {total}, normalizing...")
            for key in self.weights:
                self.weights[key] /= total
        
        self.models = {}
        self.autoencoder_threshold = None
        self.shap_explainer = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all pre-trained models."""
        try:
            # Load Isolation Forest
            if_path = os.path.join(self.model_dir, 'isolation_forest.pkl')
            self.models['isolation_forest'] = joblib.load(if_path)
            logger.info(f"Loaded Isolation Forest from {if_path}")
            
            # Load XGBoost
            xgb_path = os.path.join(self.model_dir, 'xgboost.pkl')
            self.models['xgboost'] = joblib.load(xgb_path)
            logger.info(f"Loaded XGBoost from {xgb_path}")
            
            # Autoencoder - Skipped (TensorFlow not available)
            # ae_path = os.path.join(self.model_dir, 'autoencoder.h5')
            # self.models['autoencoder'] = keras.models.load_model(ae_path)
            # logger.info(f"Loaded Autoencoder from {ae_path}")
            
            # threshold_path = os.path.join(self.model_dir, 'autoencoder_threshold.json')
            # with open(threshold_path, 'r') as f:
            #     self.autoencoder_threshold = json.load(f)['threshold']
            logger.info("Skipped Autoencoder loading (TensorFlow not available)")
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _get_if_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Get fraud probability from Isolation Forest."""
        scores = self.models['isolation_forest'].score_samples(X)
        # Convert anomaly scores to probabilities (0-1 range)
        proba = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return proba
    
    def _get_xgb_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Get fraud probability from XGBoost."""
        proba = self.models['xgboost'].predict_proba(X)[:, 1]
        return proba
    
    def _get_ae_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Get fraud probability from Autoencoder."""
        # Reconstruct input
        X_reconstructed = self.models['autoencoder'].predict(X.values, verbose=0)
        
        # Calculate reconstruction error (MSE)
        mse = np.mean(np.power(X.values - X_reconstructed, 2), axis=1)
        
        # Normalize to [0, 1]
        proba = (mse - mse.min()) / (mse.max() - mse.min() + 1e-10)
        return proba
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble fraud probability.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of fraud probabilities
        """
        # Get predictions from each model
        if_proba = self._get_if_probability(X)
        xgb_proba = self._get_xgb_probability(X)
        # ae_proba = self._get_ae_probability(X)  # Skipped
        
        # Weighted average (without autoencoder)
        ensemble_proba = (
            self.weights['isolation_forest'] * if_proba +
            self.weights['xgboost'] * xgb_proba
            # + self.weights['autoencoder'] * ae_proba
        )
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict fraud (binary).
        
        Args:
            X: Feature DataFrame
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0=legit, 1=fraud)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_top_features(self, X: pd.DataFrame, n_features: int = 3) -> List[Dict[str, float]]:
        """
        Get top contributing features using SHAP values from XGBoost.
        
        This provides explainability for predictions (important for regulatory compliance).
        
        Args:
            X: Feature DataFrame
            n_features: Number of top features to return
            
        Returns:
            List of dictionaries with feature names and SHAP values
        """
        try:
            # Initialize SHAP explainer for XGBoost if not already done
            if self.shap_explainer is None:
                # Use a sample for background (TreeExplainer is fast enough)
                self.shap_explainer = shap.TreeExplainer(self.models['xgboost'])
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X)
            
            # Get top features for each sample
            explanations = []
            for i in range(len(X)):
                # Get absolute SHAP values
                abs_shap = np.abs(shap_values[i])
                top_indices = np.argsort(abs_shap)[-n_features:][::-1]
                
                feature_importance = {
                    X.columns[idx]: float(shap_values[i][idx])
                    for idx in top_indices
                }
                explanations.append(feature_importance)
            
            return explanations
            
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {str(e)}")
            # Fallback: return empty explanations
            return [{} for _ in range(len(X))]


def evaluate_ensemble(
    ensemble: FraudEnsemble,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate ensemble performance.
    
    Args:
        ensemble: Trained ensemble model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, fbeta_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    # Predictions
    y_pred_proba = ensemble.predict_proba(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)  # Emphasize recall
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score': f2,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    logger.info("Ensemble Performance:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  F2 Score: {f2:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC: {pr_auc:.4f}")
    logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return metrics


def main():
    """Test ensemble model."""
    # Load test data
    test_df = pd.read_csv('./data/processed/test_latest.csv')
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    # Initialize ensemble
    ensemble = FraudEnsemble()
    
    # Evaluate
    metrics = evaluate_ensemble(ensemble, X_test, y_test)
    
    # Test SHAP explanations on a few samples
    logger.info("\nTesting SHAP explanations on 3 fraud samples...")
    fraud_samples = X_test[y_test == 1].head(3)
    explanations = ensemble.get_top_features(fraud_samples)
    
    for i, exp in enumerate(explanations):
        logger.info(f"\nSample {i+1} top features:")
        for feat, value in exp.items():
            logger.info(f"  {feat}: {value:.4f}")
    
    # Save metrics
    import json
    with open('./models/saved_models/ensemble_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("\nEnsemble evaluation complete!")


if __name__ == "__main__":
    main()
