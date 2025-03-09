#!/usr/bin/env python3
"""
Demo of the purified TerabyteFeatureExtractor.

This script demonstrates how to use the purified version of TerabyteFeatureExtractor
with different backends.
"""

import os
import sys
import logging
import pandas as pd

# Add parent directory to path to import emberharmony
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('purified_feature_extractor_demo')

# Import backend utilities and ops
from ember_ml.utils import backend_utils
from ember_ml import ops

# Import the TerabyteFeatureExtractor (now using the purified backend-agnostic implementation)
from ember_ml.features.terabyte_feature_extractor import TerabyteFeatureExtractor

def create_sample_data(num_rows=1000):
    """Create sample data for demonstration."""
    # Set random seed using backend-agnostic approach
    backend_utils.initialize_random_seed(42)
    
    # Generate random data using backend-agnostic operations
    numeric1 = backend_utils.tensor_to_numpy_safe(ops.random_normal(shape=(num_rows,)))
    numeric2 = backend_utils.tensor_to_numpy_safe(ops.random_normal(shape=(num_rows,)))
    
    # For categorical data, we still need to use pandas/numpy directly
    # but we can minimize direct numpy usage
    import numpy as np  # Local import only where needed
    category1_idx = backend_utils.tensor_to_numpy_safe(
        ops.random_uniform(shape=(num_rows,), minval=0, maxval=3).astype(ops.int32)
    )
    category2_idx = backend_utils.tensor_to_numpy_safe(
        ops.random_uniform(shape=(num_rows,), minval=0, maxval=3).astype(ops.int32)
    )
    boolean1 = backend_utils.tensor_to_numpy_safe(
        ops.random_uniform(shape=(num_rows,), minval=0, maxval=2).astype(ops.int32)
    ) > 0
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'numeric1': numeric1,
        'numeric2': numeric2,
        'category1': [['A', 'B', 'C'][i] for i in category1_idx],
        'category2': [['X', 'Y', 'Z'][i] for i in category2_idx],
        'datetime1': pd.date_range(start='2023-01-01', periods=num_rows, freq='H'),
        'boolean1': boolean1
    })
    
    # Create target variable using backend-agnostic operations
    noise = backend_utils.tensor_to_numpy_safe(ops.random_normal(shape=(num_rows,)) * 0.5)
    df['target'] = 2 * df['numeric1'] - 3 * df['numeric2'] + noise
    
    return df

def mock_bigquery_read(query):
    """Mock BigQuery read function for demonstration."""
    # Create sample data
    df = create_sample_data()
    
    # Print the query that would be executed
    logger.info(f"Would execute query: {query}")
    
    return df

def run_demo():
    """Run the demonstration."""
    # Print backend information
    logger.info("Backend information:")
    backend_utils.print_backend_info()
    
    # Try different backends
    backends = ['numpy', 'mlx', 'torch']
    
    for backend_name in backends:
        try:
            logger.info(f"\n\n--- Testing with {backend_name} backend ---")
            
            # Create extractor with specified backend
            extractor = TerabyteFeatureExtractor(
                project_id="demo-project",
                location="US",
                chunk_size=1000,
                max_memory_gb=4.0,
                preferred_backend=backend_name
            )
            
            # Patch BigQuery read function for demonstration
            extractor.process_bigquery_in_chunks = lambda *args, **kwargs: mock_bigquery_read(
                extractor.optimize_bigquery_query(*args, **kwargs)
            )
            
            # Prepare data
            logger.info("Preparing data...")
            result = extractor.prepare_data(
                table_id="demo-dataset.demo-table",
                target_column="target",
                force_categorical_columns=["category1", "category2"],
                limit=1000
            )
            
            if result:
                train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer = result
                
                logger.info(f"Train shape: {train_df.shape}")
                logger.info(f"Validation shape: {val_df.shape}")
                logger.info(f"Test shape: {test_df.shape}")
                logger.info(f"Number of features: {len(train_features)}")
                logger.info(f"Features: {train_features[:5]}...")  # Show first 5 features
                
                # Demonstrate backend-specific operations
                if backend_name != 'numpy':
                    logger.info(f"\nDemonstrating {backend_name} backend operations:")
                    
                    # Convert train data to tensor
                    train_tensor = backend_utils.convert_to_tensor_safe(train_df[train_features].values)
                    
                    # Perform some operations
                    mean = backend_utils.tensor_to_numpy_safe(train_tensor.mean(axis=0))
                    std = backend_utils.tensor_to_numpy_safe(train_tensor.std(axis=0))
                    
                    logger.info(f"Mean of first 5 features: {mean[:5]}")
                    logger.info(f"Std of first 5 features: {std[:5]}")
            else:
                logger.error("Failed to prepare data")
        
        except Exception as e:
            logger.error(f"Error with {backend_name} backend: {e}")
            import traceback
            logger.error(traceback.format_exc())

def main():
    """Main function."""
    logger.info("Running purified TerabyteFeatureExtractor demo")
    run_demo()
    logger.info("Demo completed")

if __name__ == '__main__':
    main()