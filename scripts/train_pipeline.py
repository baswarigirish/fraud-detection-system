"""
End-to-end training pipeline.

Runs data preprocessing and trains all models in one go.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import preprocess_pipeline
from models.train import main as train_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_full_pipeline(data_path: str = './data/raw/creditcard.csv'):
    """
    Run complete training pipeline.
    
    Steps:
    1. Preprocess data
    2. Train all models (IF, XGB, AE)
    3. Evaluate ensemble
    
    Args:
        data_path: Path to raw dataset
    """
    logger.info("\n" + "="*70)
    logger.info("FRAUD DETECTION SYSTEM - FULL TRAINING PIPELINE")
    logger.info("="*70 + "\n")
    
    # Step 1: Preprocessing
    logger.info("Step 1/3: Preprocessing data...")
    try:
        metadata = preprocess_pipeline(data_path)
        logger.info(f"✓ Preprocessing complete. Processed {metadata['train_size']} train samples.")
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {str(e)}")
        return False
    
    # Step 2: Model Training
    logger.info("\nStep 2/3: Training models...")
    try:
        train_models()
        logger.info("✓ Model training complete.")
    except Exception as e:
        logger.error(f"✗ Model training failed: {str(e)}")
        return False
    
    # Step 3: Evaluation
    logger.info("\nStep 3/3: Evaluating ensemble...")
    try:
        from models.evaluate import main as evaluate_models
        evaluate_models()
        logger.info("✓ Evaluation complete.")
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {str(e)}")
        return False
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info("\nNext steps:")
    logger.info("  - View results: ls outputs/evaluation/")
    logger.info("  - Start API: uvicorn api.main:app --reload")
    logger.info("  - View MLflow: mlflow ui --port 5000")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full training pipeline')
    parser.add_argument(
        '--data',
        default='./data/raw/creditcard.csv',
        help='Path to raw dataset'
    )
    
    args = parser.parse_args()
    
    success = run_full_pipeline(args.data)
    sys.exit(0 if success else 1)
