"""
Script to download the Kaggle Credit Card Fraud Detection dataset.

Requirements:
1. Kaggle account
2. Kaggle API credentials (~/.kaggle/kaggle.json)
3. kaggle package installed (pip install kaggle)

Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
"""

import os
import sys
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_kaggle_api():
    """Check if Kaggle API is configured."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        logger.error("Kaggle API credentials not found!")
        logger.info("\nTo set up Kaggle API:")
        logger.info("1. Go to https://www.kaggle.com/account")
        logger.info("2. Click 'Create New API Token'")
        logger.info("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        logger.info("4. On Linux/Mac: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    logger.info("Kaggle API credentials found ✓")
    return True


def download_dataset(output_dir: str = './data/raw'):
    """
    Download the Credit Card Fraud Detection dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import kaggle
        
        logger.info("Downloading Credit Card Fraud Detection dataset...")
        logger.info("This may take a few minutes (150MB file)")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=output_dir,
            unzip=True
        )
        
        logger.info(f"✓ Dataset downloaded to {output_dir}")
        
        # Verify file exists
        csv_path = os.path.join(output_dir, 'creditcard.csv')
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
            logger.info(f"✓ Found creditcard.csv ({file_size:.2f} MB)")
            return True
        else:
            logger.error("✗ creditcard.csv not found after download")
            return False
            
    except ImportError:
        logger.error("Kaggle package not installed!")
        logger.info("Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("Kaggle Credit Card Fraud Detection Dataset Downloader")
    print("="*60 + "\n")
    
    # Check Kaggle API setup
    if not setup_kaggle_api():
        sys.exit(1)
    
    # Download dataset
    success = download_dataset()
    
    if success:
        print("\n✓ Download complete!")
        print("\nNext steps:")
        print("1. Run data preprocessing: python data/preprocessing.py")
        print("2. Or explore the data: jupyter notebook notebooks/01_eda.ipynb")
    else:
        print("\n✗ Download failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
