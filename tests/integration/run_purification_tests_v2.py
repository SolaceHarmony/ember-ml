#!/usr/bin/env python3
"""
Run tests and benchmarks for the purified TerabyteFeatureExtractor and TerabyteTemporalStrideProcessor.

This script:
1. Runs the unit tests to verify the purified implementation
2. Demonstrates using both the original and purified implementations
3. Benchmarks the performance of both implementations on a larger dataset

Usage:
    python run_purification_tests_v2.py
"""

import unittest
import pandas as pd
import numpy as np
import time
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('purification_tests')

# Import the original implementation (for comparison)
from ember_ml.features.terabyte_feature_extractor import (
    TerabyteFeatureExtractor,
    TerabyteTemporalStrideProcessor
)

# Alias the classes for clarity in the tests
OriginalExtractor = TerabyteFeatureExtractor
OriginalProcessor = TerabyteTemporalStrideProcessor
PurifiedExtractor = TerabyteFeatureExtractor
PurifiedProcessor = TerabyteTemporalStrideProcessor

# Import backend utilities
from ember_ml.utils import backend_utils


def run_tests():
    """Run the unit tests for the purified implementation."""
    logger.info("Running unit tests for purified implementation...")
    from tests.test_terabyte_feature_extractor_purified_v2 import TestTerabyteFeatureExtractorPurifiedV2
    
    # Create a test suite with the test case
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTerabyteFeatureExtractorPurifiedV2)
    
    # Run the tests
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Check if all tests passed
    if test_result.wasSuccessful():
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed!")
    
    return test_result.wasSuccessful()


def demonstrate_usage():
    """Demonstrate using both the original and purified implementations."""
    logger.info("Demonstrating usage of original and purified implementations...")
    
    # Print backend information
    backend_utils.print_backend_info()
    
    # Create sample data
    np.random.seed(42)  # For reproducibility
    sample_data = pd.DataFrame({
        'numeric_col1': np.random.randn(1000),
        'numeric_col2': np.random.randn(1000),
        'categorical_col': np.random.choice(['A', 'B', 'C'], size=1000),
        'datetime_col': pd.date_range(start='2023-01-01', periods=1000, freq='H')
    })
    
    # Demonstrate original implementation
    logger.info("Using original implementation:")
    original_extractor = OriginalExtractor(
        project_id="test-project",
        location="US",
        chunk_size=100,
        max_memory_gb=1.0,
        verbose=False
    )
    
    # Create datetime features with original implementation
    start_time = time.time()
    original_result = original_extractor._create_datetime_features(sample_data[['datetime_col']].copy(), 'datetime_col')
    original_time = time.time() - start_time
    logger.info(f"Original implementation took {original_time:.4f} seconds")
    logger.info(f"Original result columns: {original_result.columns.tolist()}")
    
    # Demonstrate purified implementation
    logger.info("\nUsing purified implementation:")
    purified_extractor = PurifiedExtractor(
        project_id="test-project",
        location="US",
        chunk_size=100,
        max_memory_gb=1.0,
        verbose=False,
        preferred_backend=None  # Use default backend
    )
    
    # Create datetime features with purified implementation
    start_time = time.time()
    purified_result = purified_extractor._create_datetime_features(sample_data[['datetime_col']].copy(), 'datetime_col')
    purified_time = time.time() - start_time
    logger.info(f"Purified implementation took {purified_time:.4f} seconds")
    logger.info(f"Purified result columns: {purified_result.columns.tolist()}")
    
    # Compare performance
    speedup = original_time / purified_time
    logger.info(f"\nSpeedup: {speedup:.2f}x")
    
    # Demonstrate temporal stride processor
    logger.info("\nDemonstrating TerabyteTemporalStrideProcessor:")
    
    # Create sample data for temporal stride processing
    data = np.random.randn(100, 5)
    
    # Create original processor
    original_processor = OriginalProcessor(
        window_size=3,
        stride_perspectives=[1, 2],
        pca_components=2,
        batch_size=100,
        use_incremental_pca=False,
        verbose=False
    )
    
    # Process data with original implementation
    start_time = time.time()
    original_result = original_processor.process_batch(data)
    original_time = time.time() - start_time
    logger.info(f"Original implementation took {original_time:.4f} seconds")
    logger.info(f"Original result strides: {list(original_result.keys())}")
    for stride, result in original_result.items():
        logger.info(f"  Stride {stride} shape: {result.shape}")
    
    # Create purified processor
    purified_processor = PurifiedProcessor(
        window_size=3,
        stride_perspectives=[1, 2],
        pca_components=2,
        batch_size=100,
        use_incremental_pca=False,
        verbose=False,
        preferred_backend=None  # Use default backend
    )
    
    # Process data with purified implementation
    start_time = time.time()
    purified_result = purified_processor.process_batch(backend_utils.convert_to_tensor_safe(data))
    purified_time = time.time() - start_time
    logger.info(f"Purified implementation took {purified_time:.4f} seconds")
    logger.info(f"Purified result strides: {list(purified_result.keys())}")
    for stride, result in purified_result.items():
        result_np = backend_utils.tensor_to_numpy_safe(result)
        logger.info(f"  Stride {stride} shape: {result_np.shape}")
    
    # Compare performance
    speedup = original_time / purified_time
    logger.info(f"\nSpeedup: {speedup:.2f}x")


def benchmark_performance(size=10000, iterations=5):
    """
    Benchmark the performance of both implementations on a larger dataset.
    
    Args:
        size: Size of the dataset to use for benchmarking
        iterations: Number of iterations to run for each benchmark
    """
    logger.info(f"Benchmarking performance with dataset size {size} and {iterations} iterations...")
    
    # Create a larger dataset for benchmarking
    np.random.seed(42)  # For reproducibility
    benchmark_data = pd.DataFrame({
        'numeric_col1': np.random.randn(size),
        'numeric_col2': np.random.randn(size),
        'categorical_col': np.random.choice(['A', 'B', 'C'], size=size),
        'datetime_col': pd.date_range(start='2023-01-01', periods=size, freq='H')
    })
    
    # Benchmark datetime feature creation
    logger.info("\nBenchmarking datetime feature creation:")
    
    # Original implementation
    original_extractor = OriginalExtractor(
        project_id="test-project",
        location="US",
        chunk_size=1000,
        max_memory_gb=1.0,
        verbose=False
    )
    
    original_times = []
    for i in range(iterations):
        start_time = time.time()
        original_extractor._create_datetime_features(benchmark_data[['datetime_col']].copy(), 'datetime_col')
        original_times.append(time.time() - start_time)
    
    original_avg_time = sum(original_times) / len(original_times)
    logger.info(f"Original implementation average time: {original_avg_time:.4f} seconds")
    
    # Purified implementation
    purified_extractor = PurifiedExtractor(
        project_id="test-project",
        location="US",
        chunk_size=1000,
        max_memory_gb=1.0,
        verbose=False,
        preferred_backend=None  # Use default backend
    )
    
    purified_times = []
    for i in range(iterations):
        start_time = time.time()
        purified_extractor._create_datetime_features(benchmark_data[['datetime_col']].copy(), 'datetime_col')
        purified_times.append(time.time() - start_time)
    
    purified_avg_time = sum(purified_times) / len(purified_times)
    logger.info(f"Purified implementation average time: {purified_avg_time:.4f} seconds")
    
    # Compare performance
    speedup = original_avg_time / purified_avg_time
    logger.info(f"Speedup: {speedup:.2f}x")
    
    # Benchmark temporal stride processing
    logger.info("\nBenchmarking temporal stride processing:")
    
    # Create data for temporal stride processing
    stride_data = np.random.randn(size // 10, 10)  # Smaller data for PCA
    
    # Original implementation
    original_processor = OriginalProcessor(
        window_size=5,
        stride_perspectives=[1, 3, 5],
        pca_components=5,
        batch_size=1000,
        use_incremental_pca=True,
        verbose=False
    )
    
    original_times = []
    for i in range(iterations):
        original_processor.pca_models = {}  # Reset PCA models
        start_time = time.time()
        original_processor.process_batch(stride_data)
        original_times.append(time.time() - start_time)
    
    original_avg_time = sum(original_times) / len(original_times)
    logger.info(f"Original implementation average time: {original_avg_time:.4f} seconds")
    
    # Purified implementation
    purified_processor = PurifiedProcessor(
        window_size=5,
        stride_perspectives=[1, 3, 5],
        pca_components=5,
        batch_size=1000,
        use_incremental_pca=True,
        verbose=False,
        preferred_backend=None  # Use default backend
    )
    
    purified_times = []
    for i in range(iterations):
        purified_processor.pca_models = {}  # Reset PCA models
        start_time = time.time()
        purified_processor.process_batch(backend_utils.convert_to_tensor_safe(stride_data))
        purified_times.append(time.time() - start_time)
    
    purified_avg_time = sum(purified_times) / len(purified_times)
    logger.info(f"Purified implementation average time: {purified_avg_time:.4f} seconds")
    
    # Compare performance
    speedup = original_avg_time / purified_avg_time
    logger.info(f"Speedup: {speedup:.2f}x")
    
    # Try different backends if available
    try:
        from ember_ml.backend import set_backend, get_backend
        
        available_backends = []
        
        # Check if MLX is available
        try:
            set_backend('mlx')
            if get_backend() == 'mlx':
                available_backends.append('mlx')
        except:
            pass
        
        # Check if PyTorch is available
        try:
            set_backend('torch')
            if get_backend() == 'torch':
                available_backends.append('torch')
        except:
            pass
        
        # Always include NumPy
        set_backend('numpy')
        available_backends.append('numpy')
        
        if len(available_backends) > 1:
            logger.info("\nBenchmarking with different backends:")
            
            for backend in available_backends:
                logger.info(f"\nUsing {backend} backend:")
                set_backend(backend)
                
                purified_processor = PurifiedProcessor(
                    window_size=5,
                    stride_perspectives=[1, 3, 5],
                    pca_components=5,
                    batch_size=1000,
                    use_incremental_pca=True,
                    verbose=False,
                    preferred_backend=backend
                )
                
                purified_times = []
                for i in range(iterations):
                    purified_processor.pca_models = {}  # Reset PCA models
                    start_time = time.time()
                    purified_processor.process_batch(backend_utils.convert_to_tensor_safe(stride_data))
                    purified_times.append(time.time() - start_time)
                
                purified_avg_time = sum(purified_times) / len(purified_times)
                logger.info(f"{backend} backend average time: {purified_avg_time:.4f} seconds")
                
                # Compare performance with original
                speedup = original_avg_time / purified_avg_time
                logger.info(f"Speedup vs. original: {speedup:.2f}x")
    except ImportError:
        logger.warning("Could not import emberharmony.backend for backend comparison")


def main():
    """Main function to run tests, demonstrate usage, and benchmark performance."""
    parser = argparse.ArgumentParser(description='Run tests and benchmarks for purified implementation')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running unit tests')
    parser.add_argument('--skip-demo', action='store_true', help='Skip demonstration of usage')
    parser.add_argument('--skip-benchmark', action='store_true', help='Skip performance benchmarks')
    parser.add_argument('--benchmark-size', type=int, default=10000, help='Size of dataset for benchmarking')
    parser.add_argument('--benchmark-iterations', type=int, default=5, help='Number of iterations for benchmarking')
    
    args = parser.parse_args()
    
    # Print header
    logger.info("=" * 80)
    logger.info("TerabyteFeatureExtractor Purification Tests")
    logger.info("=" * 80)
    
    # Run tests if not skipped
    if not args.skip_tests:
        logger.info("\n" + "=" * 80)
        logger.info("Running Unit Tests")
        logger.info("=" * 80)
        tests_passed = run_tests()
        if not tests_passed:
            logger.error("Unit tests failed! Exiting...")
            return 1
    
    # Demonstrate usage if not skipped
    if not args.skip_demo:
        logger.info("\n" + "=" * 80)
        logger.info("Demonstrating Usage")
        logger.info("=" * 80)
        demonstrate_usage()
    
    # Benchmark performance if not skipped
    if not args.skip_benchmark:
        logger.info("\n" + "=" * 80)
        logger.info("Benchmarking Performance")
        logger.info("=" * 80)
        benchmark_performance(size=args.benchmark_size, iterations=args.benchmark_iterations)
    
    logger.info("\n" + "=" * 80)
    logger.info("All tests completed successfully!")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())