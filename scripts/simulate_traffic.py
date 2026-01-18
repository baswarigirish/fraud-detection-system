"""
Load testing script for the fraud detection API.

Simulates realistic traffic patterns to test API performance.
"""

import time
import random
import asyncio
import logging
from typing import List, Dict
from datetime import datetime
import statistics

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_transaction() -> Dict:
    """Generate a random transaction for testing."""
    # Generate realistic transaction features
    transaction = {
        "Time": random.randint(0, 172800),  # ~2 days in seconds
        "Amount": round(random.lognormvariate(3, 2), 2),  # Log-normal distribution
    }
    
    # Add V1-V28 features (PCA components, normally distributed)
    for i in range(1, 29):
        transaction[f"V{i}"] = round(random.gauss(0, 1), 6)
    
    return transaction


def load_real_samples(filepath: str = './data/processed/test_latest.csv', n: int = 100) -> List[Dict]:
    """Load real transaction samples from test set."""
    try:
        df = pd.read_csv(filepath)
        df = df.drop('Class', axis=1)
        samples = df.sample(min(n, len(df))).to_dict('records')
        return samples
    except FileNotFoundError:
        logger.warning(f"Test file not found: {filepath}. Using generated samples.")
        return [generate_sample_transaction() for _ in range(n)]


def test_single_prediction(api_url: str, transaction: Dict, api_key: str = None) -> Dict:
    """Test single prediction endpoint."""
    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=transaction,
            headers=headers,
            timeout=5
        )
        
        latency = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            return {
                'status': 'success',
                'latency_ms': latency,
                'response': response.json()
            }
        else:
            return {
                'status': 'error',
                'status_code': response.status_code,
                'latency_ms': latency,
                'error': response.text
            }
    except Exception as e:
        return {
            'status': 'exception',
            'error': str(e),
            'latency_ms': (time.time() - start_time) * 1000
        }


def test_batch_prediction(api_url: str, transactions: List[Dict], api_key: str = None) -> Dict:
    """Test batch prediction endpoint."""
    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{api_url}/predict/batch",
            json={"transactions": transactions},
            headers=headers,
            timeout=30
        )
        
        latency = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            return {
                'status': 'success',
                'latency_ms': latency,
                'batch_size': len(transactions),
                'avg_latency_per_item': latency / len(transactions)
            }
        else:
            return {
                'status': 'error',
                'status_code': response.status_code,
                'latency_ms': latency
            }
    except Exception as e:
        return {
            'status': 'exception',
            'error': str(e)
        }


def run_load_test(
    api_url: str = "http://localhost:8000",
    n_requests: int = 100,
    use_real_data: bool = True,
    api_key: str = None
):
    """
    Run load test on fraud detection API.
    
    Args:
        api_url: Base URL of the API
        n_requests: Number of requests to send
        use_real_data: Use real samples from test set (vs generated)
        api_key: API key for authentication
    """
    logger.info("="*70)
    logger.info("FRAUD DETECTION API - LOAD TEST")
    logger.info("="*70)
    logger.info(f"API URL: {api_url}")
    logger.info(f"Total requests: {n_requests}")
    logger.info(f"Using real data: {use_real_data}")
    logger.info("="*70 + "\n")
    
    # Check health endpoint first
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        if health.status_code == 200:
            logger.info("✓ API is healthy")
            logger.info(f"  Response: {health.json()}\n")
        else:
            logger.error("✗ API health check failed")
            return
    except Exception as e:
        logger.error(f"✗ Cannot connect to API: {str(e)}")
        return
    
    # Load or generate samples
    if use_real_data:
        samples = load_real_samples(n=n_requests)
    else:
        samples = [generate_sample_transaction() for _ in range(n_requests)]
    
    # Test single predictions
    logger.info(f"Testing {n_requests} single predictions...")
    results = []
    latencies = []
    errors = 0
    
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        result = test_single_prediction(api_url, sample, api_key)
        results.append(result)
        
        if result['status'] == 'success':
            latencies.append(result['latency_ms'])
        else:
            errors += 1
        
        # Progress update every 20 requests
        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i+1}/{n_requests} requests...")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if latencies:
        logger.info("\n" + "="*70)
        logger.info("RESULTS")
        logger.info("="*70)
        logger.info(f"Total requests: {n_requests}")
        logger.info(f"Successful: {len(latencies)}")
        logger.info(f"Failed: {errors}")
        logger.info(f"Success rate: {len(latencies)/n_requests*100:.1f}%")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Throughput: {n_requests/total_time:.1f} req/s")
        logger.info(f"\nLatency Statistics (ms):")
        logger.info(f"  Mean: {statistics.mean(latencies):.2f}")
        logger.info(f"  Median: {statistics.median(latencies):.2f}")
        logger.info(f"  P95: {np.percentile(latencies, 95):.2f}")
        logger.info(f"  P99: {np.percentile(latencies, 99):.2f}")
        logger.info(f"  Min: {min(latencies):.2f}")
        logger.info(f"  Max: {max(latencies):.2f}")
        
        # Check if target latency is met
        p95_latency = np.percentile(latencies, 95)
        if p95_latency < 100:
            logger.info(f"\n✓ Target latency met! P95 = {p95_latency:.2f}ms < 100ms")
        else:
            logger.warning(f"\n⚠ Target latency NOT met. P95 = {p95_latency:.2f}ms > 100ms")
    else:
        logger.error("No successful requests!")
    
    # Test batch prediction
    logger.info("\n" + "="*70)
    logger.info("Testing batch prediction (100 transactions)...")
    batch_result = test_batch_prediction(api_url, samples[:100], api_key)
    
    if batch_result['status'] == 'success':
        logger.info(f"✓ Batch prediction successful")
        logger.info(f"  Total latency: {batch_result['latency_ms']:.2f}ms")
        logger.info(f"  Avg per transaction: {batch_result['avg_latency_per_item']:.2f}ms")
    else:
        logger.error(f"✗ Batch prediction failed: {batch_result.get('error', 'Unknown error')}")
    
    logger.info("\n" + "="*70)
    logger.info("LOAD TEST COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load test fraud detection API')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests')
    parser.add_argument('--generated', action='store_true', help='Use generated data instead of real')
    parser.add_argument('--api-key', default=None, help='API key for authentication')
    
    args = parser.parse_args()
    
    run_load_test(
        api_url=args.url,
        n_requests=args.requests,
        use_real_data=not args.generated,
        api_key=args.api_key
    )
