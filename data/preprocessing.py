"""
Data preprocessing pipeline for fraud detection.

This module handles loading, feature engineering, and preparing data for training.
Tried regular oversampling first but SMOTE worked better for this imbalanced dataset.
"""

import os
import json
import logging
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with transaction data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or corrupted
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} transactions from {filepath}")
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Log basic stats
        fraud_count = df['Class'].sum()
        fraud_rate = fraud_count / len(df) * 100
        logger.info(f"Fraud transactions: {fraud_count} ({fraud_rate:.4f}%)")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from existing ones.
    
    Features added:
    - Transaction hour (from Time)
    - Amount z-score and log transform
    - Rolling statistics (mean/std) - simulated with time bins
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Extract hour from Time (seconds since first transaction)
    # Assuming Time represents seconds in a day cycle
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Hour'] = df['Hour'].astype(int)
    
    # Log transform of Amount (handle zero values)
    df['Amount_Log'] = np.log1p(df['Amount'])
    
    # Amount z-score (temporary, will be scaled later)
    df['Amount_Zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    
    # Rolling statistics - simulate by time bins
    # Could optimize rolling stats with window functions later
    df['Time_Bin'] = pd.cut(df['Time'], bins=100, labels=False)
    
    # Calculate mean and std for each time bin
    time_stats = df.groupby('Time_Bin')['Amount'].agg(['mean', 'std']).reset_index()
    time_stats.columns = ['Time_Bin', 'Time_Bin_Amount_Mean', 'Time_Bin_Amount_Std']
    
    df = df.merge(time_stats, on='Time_Bin', how='left')
    df['Time_Bin_Amount_Std'].fillna(0, inplace=True)
    
    # Drop temporary column
    df.drop('Time_Bin', axis=1, inplace=True)
    
    logger.info(f"Engineered features. New shape: {df.shape}")
    
    return df


def split_data(
    df: pd.DataFrame, 
    train_size: float = 0.6,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        df: Full dataset
        train_size: Proportion for training (0.6 = 60%)
        val_size: Proportion for validation (0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Calculate test size
    test_size = 1.0 - train_size - val_size
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=df['Class']
    )
    
    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        stratify=temp_df['Class']
    )
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Fraud rates - Train: {train_df['Class'].mean():.4f}, "
                f"Val: {val_df['Class'].mean():.4f}, Test: {test_df['Class'].mean():.4f}")
    
    return train_df, val_df, test_df


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, 
                sampling_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to training data only.
    
    Keep validation/test imbalanced to maintain realistic distributions.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Target ratio of minority class (0.5 = 50% fraud)
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    logger.info(f"Before SMOTE - Fraud: {y_train.sum()}, Legit: {len(y_train) - y_train.sum()}")
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"After SMOTE - Fraud: {y_resampled.sum()}, Legit: {len(y_resampled) - y_resampled.sum()}")
    
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def preprocess_pipeline(
    input_path: str,
    output_dir: str = './data/processed',
    apply_smote_flag: bool = True
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline from raw data to train/val/test sets.
    
    Steps:
    1. Load raw data
    2. Engineer features
    3. Split into train/val/test
    4. Apply SMOTE to training data (optional)
    5. Fit scaler on training data, transform all sets
    6. Save processed data and artifacts
    
    Args:
        input_path: Path to raw CSV file
        output_dir: Directory to save processed data
        apply_smote_flag: Whether to apply SMOTE oversampling
        
    Returns:
        Dictionary with processing metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load and engineer features
    df = load_data(input_path)
    df = engineer_features(df)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    train_idx, temp_idx = train_test_split(
        range(len(df)), test_size=0.4, random_state=42, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=y.iloc[temp_idx]
    )
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    # Apply SMOTE to training data only
    if apply_smote_flag:
        X_train, y_train = apply_smote(X_train, y_train)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save processed data
    train_df = X_train_scaled.copy()
    train_df['Class'] = y_train.values
    train_df.to_csv(f"{output_dir}/train_{timestamp}.csv", index=False)
    
    val_df = X_val_scaled.copy()
    val_df['Class'] = y_val.values
    val_df.to_csv(f"{output_dir}/val_{timestamp}.csv", index=False)
    
    test_df = X_test_scaled.copy()
    test_df['Class'] = y_test.values
    test_df.to_csv(f"{output_dir}/test_{timestamp}.csv", index=False)
    
    # Also save latest versions without timestamp
    train_df.to_csv(f"{output_dir}/train_latest.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_latest.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_latest.csv", index=False)
    
    # Save scaler
    scaler_path = f"{output_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    with open(f"{output_dir}/feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Metadata
    metadata = {
        'timestamp': timestamp,
        'input_path': input_path,
        'output_dir': output_dir,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_fraud_rate': float(y_train.mean()),
        'val_fraud_rate': float(y_val.mean()),
        'test_fraud_rate': float(y_test.mean()),
        'smote_applied': apply_smote_flag
    }
    
    with open(f"{output_dir}/metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Preprocessing complete!")
    logger.info(f"Saved {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    return metadata


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "./data/raw/creditcard.csv"
    
    try:
        metadata = preprocess_pipeline(input_file)
        print("\nPreprocessing Summary:")
        print(json.dumps(metadata, indent=2))
    except FileNotFoundError:
        print(f"\nError: Dataset not found at {input_file}")
        print("Please download the Kaggle Credit Card Fraud dataset first.")
        print("Run: python scripts/download_data.py")
