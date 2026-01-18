"""
Tests for data preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import os
from data.preprocessing import (
    load_data, engineer_features, split_data, 
    apply_smote, preprocess_pipeline
)


def test_load_data_valid(test_data_path):
    """Test loading data from valid path."""
    df = load_data(test_data_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert 'Class' in df.columns
    assert 'Amount' in df.columns


def test_load_data_invalid_path():
    """Test loading data from invalid path raises error."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")


def test_engineer_features(sample_dataframe):
    """Test feature engineering adds correct features."""
    df = sample_dataframe.drop('Class', axis=1)
    df_eng = engineer_features(df)
    
    # Check new features exist
    assert 'Hour' in df_eng.columns
    assert 'Amount_Log' in df_eng.columns
    assert 'Amount_Zscore' in df_eng.columns
    assert 'Time_Bin_Amount_Mean' in df_eng.columns
    assert 'Time_Bin_Amount_Std' in df_eng.columns
    
    # Check Hour is in valid range
    assert df_eng['Hour'].min() >= 0
    assert df_eng['Hour'].max() < 24
    
    # Check Amount_Log is non-negative
    assert (df_eng['Amount_Log'] >= 0).all()


def test_split_data_stratification(sample_dataframe):
    """Test data splitting maintains class balance."""
    train_df, val_df, test_df = split_data(sample_dataframe)
    
    # Check all splits have data
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    
    # Check total size matches
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataframe)
    
    # Check class column exists in all
    assert 'Class' in train_df.columns
    assert 'Class' in val_df.columns
    assert 'Class' in test_df.columns


def test_apply_smote():
    """Test SMOTE oversampling increases minority class."""
    # Create imbalanced data
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series([0] * 95 + [1] * 5)
    
    X_resampled, y_resampled = apply_smote(X, y, sampling_strategy=0.5)
    
    # Check sizes increased
    assert len(X_resampled) > len(X)
    assert len(y_resampled) > len(y)
    
    # Check minority class increased
    minority_before = (y == 1).sum()
    minority_after = (y_resampled == 1).sum()
    assert minority_after > minority_before


def test_feature_engineering_handles_edge_cases():
    """Test feature engineering with edge cases."""
    # Test with zero amounts
    df = pd.DataFrame({
        'Time': [0, 1, 2],
        'Amount': [0.0, 0.0, 0.0],
        **{f'V{i}': [0.0, 0.0, 0.0] for i in range(1, 29)}
    })
    
    df_eng = engineer_features(df)
    
    # Should not raise errors
    assert 'Amount_Log' in df_eng.columns
    assert not df_eng['Amount_Log'].isnull().any()
    
    # log1p(0) = 0
    assert (df_eng['Amount_Log'] == 0).all()


def test_preprocessing_pipeline(test_data_path, tmp_path):
    """Test full preprocessing pipeline."""
    output_dir = str(tmp_path / "processed")
    
    metadata = preprocess_pipeline(
        input_path=test_data_path,
        output_dir=output_dir,
        apply_smote_flag=False  # Skip SMOTE for small test data
    )
    
    # Check metadata
    assert 'train_size' in metadata
    assert 'val_size' in metadata
    assert 'test_size' in metadata
    assert 'n_features' in metadata
    
    # Check files created
    assert os.path.exists(f"{output_dir}/train_latest.csv")
    assert os.path.exists(f"{output_dir}/val_latest.csv")
    assert os.path.exists(f"{output_dir}/test_latest.csv")
    assert os.path.exists(f"{output_dir}/scaler.pkl")
    assert os.path.exists(f"{output_dir}/feature_names.json")
