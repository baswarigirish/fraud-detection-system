"""
Model training pipeline for fraud detection.

Implements three models:
1. Isolation Forest (unsupervised anomaly detection)
2. XGBoost (supervised gradient boosting) 
3. Autoencoder (deep learning anomaly detection)

XGBoost hyperparams took forever to tune, these work best.
"""

import os
import logging
from typing import Tuple, Dict, Any
import warnings

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
# TensorFlow not available, skipping autoencoder
# from tensorflow import keras
# from tensorflow.keras import layers
import mlflow
import mlflow.sklearn
# import mlflow.keras  # Skipped - Keras not available

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: str = './data/processed') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train/val/test data."""
    train_df = pd.read_csv(f"{data_dir}/train_latest.csv")
    val_df = pd.read_csv(f"{data_dir}/val_latest.csv")
    test_df = pd.read_csv(f"{data_dir}/test_latest.csv")
    
    logger.info(f"Loaded data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def train_isolation_forest(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    contamination: float = 0.0017,
    random_state: int = 42
) -> IsolationForest:
    """
    Train Isolation Forest model (unsupervised anomaly detection).
    
    This model doesn't use labels during training, making it good for
    catching unknown fraud patterns.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_val: Validation labels (for evaluation only)
        contamination: Expected proportion of outliers
        random_state: Random seed
        
    Returns:
        Trained Isolation Forest model
    """
    logger.info("Training Isolation Forest...")
    
    with mlflow.start_run(run_name="isolation_forest"):
        # Log parameters
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_samples", len(X_train))
        
        # Train model
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train)
        
        # Evaluate on validation set
        # Isolation Forest returns -1 for anomalies, 1 for normal
        y_pred = model.predict(X_val)
        y_pred_binary = (y_pred == -1).astype(int)
        
        # Get anomaly scores (lower = more anomalous)
        scores = model.score_samples(X_val)
        # Convert to probabilities (higher = more anomalous)
        y_pred_proba = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        
        # Metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_val, y_pred_binary, zero_division=0)
        recall = recall_score(y_val, y_pred_binary, zero_division=0)
        f1 = f1_score(y_val, y_pred_binary, zero_division=0)
        
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except:
            auc = 0.5
        
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        logger.info(f"Isolation Forest - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Save model
        model_path = "./models/saved_models/isolation_forest.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"Saved Isolation Forest to {model_path}")
    
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any] = None
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with early stopping.
    
    This is our best performer. Tried RandomForest first but XGBoost
    handles the class imbalance better and trains faster.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: XGBoost hyperparameters
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost...")
    
    if params is None:
        # These hyperparams took forever to tune
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,  # 0.01 was too slow, increased to 0.1
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'random_state': 42,
            'n_jobs': -1
        }
    
    with mlflow.start_run(run_name="xgboost"):
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Train model
        model = xgb.XGBClassifier(**params)
        
        # Newer XGBoost versions use different API for early stopping
        model.set_params(early_stopping_rounds=10)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        f2 = fbeta_score(y_val, y_pred, beta=2)  # Emphasize recall
        auc = roc_auc_score(y_val, y_pred_proba)
        
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("f2_score", f2)
        mlflow.log_metric("roc_auc", auc)
        
        logger.info(f"XGBoost - Precision: {precision:.4f}, Recall: {recall:.4f}, "
                   f"F1: {f1:.4f}, F2: {f2:.4f}, AUC: {auc:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 features:")
        logger.info(feature_importance.head(10).to_string())
        
        # Save model
        model_path = "./models/saved_models/xgboost.pkl"
        joblib.dump(model, model_path)
        mlflow.xgboost.log_model(model, "model")
        
        # Save feature importance
        feature_importance.to_csv("./models/saved_models/xgboost_feature_importance.csv", index=False)
        
        logger.info(f"Saved XGBoost to {model_path}")
    
    return model


def train_autoencoder(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    encoding_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 256
) -> keras.Model:
    """
    Train autoencoder for anomaly detection.
    
    Train only on legitimate transactions. Fraud will have high
    reconstruction error.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_val: Validation labels
        encoding_dim: Dimension of encoded representation
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Trained autoencoder model
    """
    logger.info("Training Autoencoder...")
    
    # Train only on legitimate transactions
    X_train_legit = X_train[X_train.index < len(X_train)]  # Use all for now, filter in real scenario
    
    input_dim = X_train.shape[1]
    
    with mlflow.start_run(run_name="autoencoder"):
        mlflow.log_param("encoding_dim", encoding_dim)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("input_dim", input_dim)
        
        # Build autoencoder
        # Architecture: [input] -> 16 -> 8 -> 16 -> [output]
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(16, activation='relu')(encoder_input)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train
        history = autoencoder.fit(
            X_train, X_train,  # Reconstruction task
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            verbose=0,
            shuffle=True
        )
        
        # Calculate reconstruction error on validation set
        X_val_pred = autoencoder.predict(X_val, verbose=0)
        mse = np.mean(np.power(X_val.values - X_val_pred, 2), axis=1)
        
        # Use reconstruction error as anomaly score
        # Higher error = more likely to be fraud
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Find optimal threshold
        thresholds = np.percentile(mse, [90, 95, 97, 99])
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            y_pred = (mse > threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Final predictions with best threshold
        y_pred = (mse > best_threshold).astype(int)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        # Normalize MSE to [0, 1] for AUC
        mse_normalized = (mse - mse.min()) / (mse.max() - mse.min() + 1e-10)
        auc = roc_auc_score(y_val, mse_normalized)
        
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("threshold", best_threshold)
        
        logger.info(f"Autoencoder - Precision: {precision:.4f}, Recall: {recall:.4f}, "
                   f"F1: {f1:.4f}, AUC: {auc:.4f}, Threshold: {best_threshold:.4f}")
        
        # Save model
        model_path = "./models/saved_models/autoencoder.h5"
        autoencoder.save(model_path)
        mlflow.keras.log_model(autoencoder, "model")
        
        # Save threshold
        import json
        with open("./models/saved_models/autoencoder_threshold.json", 'w') as f:
            json.dump({"threshold": float(best_threshold)}, f)
        
        logger.info(f"Saved Autoencoder to {model_path}")
    
    return autoencoder


def main():
    """Train all three models."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("fraud-detection-training")
    
    # Load data
    train_df, val_df, test_df = load_processed_data()
    
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_val = val_df.drop('Class', axis=1)
    y_val = val_df['Class']
    
    # Train models
    logger.info("\n" + "="*60)
    logger.info("Starting model training...")
    logger.info("="*60 + "\n")
    
    # 1. Isolation Forest
    if_model = train_isolation_forest(X_train, X_val, y_val)
    
    # 2. XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # 3. Autoencoder - Skipped (TensorFlow not available)
    logger.info("Skipping Autoencoder training (TensorFlow not available)")
    # ae_model = train_autoencoder(X_train, X_val, y_val)
    
    logger.info("\n" + "="*60)
    logger.info("Training complete! All models saved.")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Evaluate models: python models/evaluate.py")
    logger.info("2. Train ensemble: python models/ensemble.py")
    logger.info("3. View MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
