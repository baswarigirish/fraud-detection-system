"""
Inference logic for fraud detection predictions.

Handles model loading, preprocessing, caching, and prediction generation.
Redis connection pooling added after timeout issues in testing.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import time

import pandas as pd
import numpy as np
import joblib
import redis

from models.ensemble import FraudEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionService:
    """
    Service for fraud detection predictions.
    
    Handles model loading, preprocessing, caching, and predictions.
    """
    
    def __init__(
        self,
        model_dir: str = './models/saved_models',
        scaler_path: str = './data/processed/scaler.pkl',
        cache_ttl: int = 300,  # 5 minutes
        redis_url: str = None
    ):
        """
        Initialize fraud detection service.
        
        Args:
            model_dir: Directory containing trained models
            scaler_path: Path to fitted scaler
            cache_ttl: Cache time-to-live in seconds
            redis_url: Redis connection URL (None to disable caching)
        """
        self.model_dir = model_dir
        self.scaler_path = scaler_path
        self.cache_ttl = cache_ttl
        self.redis_url = redis_url
        
        # Load models and scaler
        self.ensemble = None
        self.scaler = None
        self.feature_names = None
        self.redis_client = None
        
        self._load_artifacts()
        self._connect_redis()
    
    def _load_artifacts(self):
        """Load model ensemble and scaler."""
        try:
            # Load ensemble
            self.ensemble = FraudEnsemble(model_dir=self.model_dir)
            logger.info("✓ Loaded ensemble model")
            
            # Load scaler
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"✓ Loaded scaler from {self.scaler_path}")
            
            # Load feature names
            feature_path = os.path.join(os.path.dirname(self.scaler_path), 'feature_names.json')
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"✓ Loaded {len(self.feature_names)} feature names")
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {str(e)}")
            raise
    
    def _connect_redis(self):
        """Connect to Redis for caching."""
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                logger.info("✓ Connected to Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
                self.redis_client = None
        else:
            logger.info("Redis URL not provided. Caching disabled.")
    
    def _generate_cache_key(self, transaction: Dict) -> str:
        """Generate cache key from transaction data."""
        # Sort keys for consistent hashing
        sorted_items = sorted(transaction.items())
        transaction_str = json.dumps(sorted_items)
        key_hash = hashlib.md5(transaction_str.encode()).hexdigest()
        return f"fraud_pred:{key_hash}"
    
    def _get_from_cache(self, key: str) -> Dict:
        """Get prediction from cache."""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {str(e)}")
        
        return None
    
    def _set_cache(self, key: str, value: Dict):
        """Set prediction in cache."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                key,
                self.cache_ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")
    
    def _preprocess_transaction(self, transaction: Dict) -> pd.DataFrame:
        """
        Preprocess raw transaction to model input format.
        
        Applies the same feature engineering as training.
        """
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Feature engineering (same as preprocessing.py)
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Hour'] = df['Hour'].astype(int)
        
        df['Amount_Log'] = np.log1p(df['Amount'])
        
        df['Amount_Zscore'] = (df['Amount'] - df['Amount'].mean()) / (df['Amount'].std() + 1e-10)
        
        # For rolling stats, use placeholder values (can't calculate on single transaction)
        df['Time_Bin_Amount_Mean'] = df['Amount'].values[0]
        df['Time_Bin_Amount_Std'] = 0.0
        
        # Scale features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.scaler.feature_names_in_
        )
        
        return df_scaled
    
    def predict_single(
        self,
        transaction: Dict,
        threshold: float = 0.5,
        use_cache: bool = True
    ) -> Dict:
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction: Transaction dictionary
            threshold: Classification threshold
            use_cache: Whether to use caching
            
        Returns:
            Prediction dictionary with probability, risk level, explanation
        """
        start_time = time.time()
        
        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(transaction)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.debug("Cache hit")
                cached_result['cached'] = True
                return cached_result
        
        # Preprocess
        X = self._preprocess_transaction(transaction)
        
        # Predict
        fraud_proba = self.ensemble.predict_proba(X)[0]
        is_fraud = fraud_proba >= threshold
        
        # Determine risk level
        if fraud_proba < 0.3:
            risk_level = "LOW"
        elif fraud_proba < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Get explanation (top 3 features)
        try:
            explanations = self.ensemble.get_top_features(X, n_features=3)
            explanation = explanations[0] if explanations else {}
        except Exception as e:
            logger.warning(f"Explanation generation failed: {str(e)}")
            explanation = {}
        
        # Build result
        result = {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_proba),
            'risk_level': risk_level,
            'explanation': explanation,
            'latency_ms': (time.time() - start_time) * 1000,
            'cached': False
        }
        
        # Cache result
        if use_cache:
            self._set_cache(cache_key, result)
        
        return result
    
    def predict_batch(
        self,
        transactions: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Predict fraud for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            threshold: Classification threshold
            
        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()
        
        # Preprocess all transactions
        dfs = [self._preprocess_transaction(tx) for tx in transactions]
        X = pd.concat(dfs, ignore_index=True)
        
        # Batch prediction
        fraud_probas = self.ensemble.predict_proba(X)
        is_frauds = fraud_probas >= threshold
        
        # Risk levels
        risk_levels = []
        for proba in fraud_probas:
            if proba < 0.3:
                risk_levels.append("LOW")
            elif proba < 0.7:
                risk_levels.append("MEDIUM")
            else:
                risk_levels.append("HIGH")
        
        # Explanations (batch SHAP is faster)
        try:
            explanations = self.ensemble.get_top_features(X, n_features=3)
        except Exception as e:
            logger.warning(f"Batch explanation failed: {str(e)}")
            explanations = [{} for _ in range(len(transactions))]
        
        # Build results
        results = []
        for i in range(len(transactions)):
            result = {
                'is_fraud': bool(is_frauds[i]),
                'fraud_probability': float(fraud_probas[i]),
                'risk_level': risk_levels[i],
                'explanation': explanations[i],
                'cached': False
            }
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Batch prediction: {len(transactions)} transactions in {total_time:.2f}ms "
                   f"({total_time/len(transactions):.2f}ms avg)")
        
        return results
    
    def is_healthy(self) -> Tuple[bool, Dict]:
        """
        Check service health.
        
        Returns:
            Tuple of (is_healthy, status_dict)
        """
        status = {
            'model_loaded': self.ensemble is not None,
            'scaler_loaded': self.scaler is not None,
            'redis_connected': False,
            'features_loaded': self.feature_names is not None
        }
        
        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                status['redis_connected'] = True
            except:
                pass
        
        is_healthy = all([
            status['model_loaded'],
            status['scaler_loaded'],
            status['features_loaded']
        ])
        
        return is_healthy, status


# Global service instance (singleton pattern)
_service_instance = None


def get_service() -> FraudDetectionService:
    """Get or create fraud detection service instance."""
    global _service_instance
    
    if _service_instance is None:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        _service_instance = FraudDetectionService(redis_url=redis_url)
    
    return _service_instance
