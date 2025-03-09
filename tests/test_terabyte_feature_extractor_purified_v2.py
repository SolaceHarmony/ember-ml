"""
Tests for the purified version of TerabyteFeatureExtractor and TerabyteTemporalStrideProcessor.

This module compares the original NumPy-based implementation with the purified
emberharmony backend-agnostic implementation to ensure they produce equivalent results.
"""

import pytest
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_terabyte_feature_extractor')

# Import the original implementation
# Import the implementation (now purified)
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


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample data
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame({
        'numeric_col1': np.random.randn(1000),
        'numeric_col2': np.random.randn(1000),
        'categorical_col': np.random.choice(['A', 'B', 'C'], size=1000),
        'datetime_col': pd.date_range(start='2023-01-01', periods=1000, freq='H')
    })


@pytest.fixture
def original_extractor():
    """Create original extractor."""
    return OriginalExtractor(
        project_id="test-project",
        location="US",
        chunk_size=100,
        max_memory_gb=1.0,
        verbose=False
    )


@pytest.fixture
def purified_extractor():
    """Create purified extractor."""
    return PurifiedExtractor(
        project_id="test-project",
        location="US",
        chunk_size=100,
        max_memory_gb=1.0,
        verbose=False,
        preferred_backend=None  # Use default backend
    )


@pytest.fixture
def original_processor():
    """Create original processor."""
    return OriginalProcessor(
        window_size=3,
        stride_perspectives=[1, 2],
        pca_components=2,
        batch_size=100,
        use_incremental_pca=False,
        verbose=False
    )


@pytest.fixture
def purified_processor():
    """Create purified processor."""
    return PurifiedProcessor(
        window_size=3,
        stride_perspectives=[1, 2],
        pca_components=2,
        batch_size=100,
        use_incremental_pca=False,
        verbose=False,
        preferred_backend=None  # Use default backend
    )


class TestTerabyteFeatureExtractorPurifiedV2:
    """Test case for the purified TerabyteFeatureExtractor and TerabyteTemporalStrideProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        # Print backend information
        backend_utils.print_backend_info()

    def test_datetime_feature_creation(self, sample_data, original_extractor, purified_extractor):
        """Test that datetime feature creation produces equivalent results."""
        # Create a copy of the sample data with only the datetime column
        df = sample_data[['datetime_col']].copy()
        
        # Apply datetime feature creation with original implementation
        start_time = time.time()
        original_result = original_extractor._create_datetime_features(df.copy(), 'datetime_col')
        original_time = time.time() - start_time
        
        # Apply datetime feature creation with purified implementation
        start_time = time.time()
        purified_result = purified_extractor._create_datetime_features(df.copy(), 'datetime_col')
        purified_time = time.time() - start_time
        
        # Check that the results are equivalent
        for col in original_result.columns:
            if col != 'datetime_col':  # Skip the original column
                np.testing.assert_allclose(
                    original_result[col].values,
                    purified_result[col].values,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Column {col} values differ between original and purified implementations"
                )
        
        logger.info(f"Datetime feature creation: Original: {original_time:.4f}s, Purified: {purified_time:.4f}s")
        logger.info(f"Speedup: {original_time / purified_time:.2f}x")

    def test_data_splitting(self, sample_data, original_extractor, purified_extractor):
        """Test that data splitting produces equivalent results."""
        # Create a copy of the sample data
        df = sample_data.copy()
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        backend_utils.initialize_random_seed(42)
        
        # Apply data splitting with original implementation
        start_time = time.time()
        original_train, original_val, original_test = original_extractor._split_data(df.copy(), None)
        original_time = time.time() - start_time
        
        # Reset random seeds
        np.random.seed(42)
        backend_utils.initialize_random_seed(42)
        
        # Apply data splitting with purified implementation
        start_time = time.time()
        purified_train, purified_val, purified_test = purified_extractor._split_data(df.copy(), None)
        purified_time = time.time() - start_time
        
        # Check that the results have similar shapes
        assert len(original_train) == len(purified_train)
        assert len(original_val) == len(purified_val)
        assert len(original_test) == len(purified_test)
        
        logger.info(f"Data splitting: Original: {original_time:.4f}s, Purified: {purified_time:.4f}s")
        logger.info(f"Speedup: {original_time / purified_time:.2f}x")

    def test_strided_sequences(self, original_processor, purified_processor):
        """Test that strided sequence creation produces equivalent results."""
        # Create sample data for strided sequences
        data = np.random.randn(100, 5)
        
        # Create strided sequences with original implementation
        start_time = time.time()
        original_windows = original_processor._create_strided_sequences(data, stride=2)
        original_time = time.time() - start_time
        
        # Create strided sequences with purified implementation
        start_time = time.time()
        purified_windows = purified_processor._create_strided_sequences(
            backend_utils.convert_to_tensor_safe(data), stride=2
        )
        purified_time = time.time() - start_time
        
        # Convert purified windows to numpy for comparison
        purified_windows_np = [backend_utils.tensor_to_numpy_safe(window) for window in purified_windows]
        
        # Check that the results are equivalent
        assert len(original_windows) == len(purified_windows)
        for i in range(len(original_windows)):
            np.testing.assert_allclose(
                original_windows[i],
                purified_windows_np[i],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Window {i} differs between original and purified implementations"
            )
        
        logger.info(f"Strided sequences: Original: {original_time:.4f}s, Purified: {purified_time:.4f}s")
        logger.info(f"Speedup: {original_time / purified_time:.2f}x")

    def test_pca_blend(self, original_processor, purified_processor):
        """Test that PCA blending produces equivalent results."""
        # Create sample data for PCA blending
        window_batch = np.random.randn(50, 3, 5)
        
        # Apply PCA blending with original implementation
        start_time = time.time()
        original_result = original_processor._apply_pca_blend(window_batch, stride=1)
        original_time = time.time() - start_time
        
        # Reset PCA models to ensure fair comparison
        purified_processor.pca_models = {}
        
        # Apply PCA blending with purified implementation
        start_time = time.time()
        purified_result = purified_processor._apply_pca_blend(
            backend_utils.convert_to_tensor_safe(window_batch), stride=1
        )
        purified_time = time.time() - start_time
        
        # Convert purified result to numpy for comparison
        purified_result_np = backend_utils.tensor_to_numpy_safe(purified_result)
        
        # Check that the results have the same shape
        assert original_result.shape == purified_result_np.shape
        
        # Check that the feature importance calculations are equivalent
        original_importance = original_processor.get_feature_importance(1)
        purified_importance = purified_processor.get_feature_importance(1)
        purified_importance_np = backend_utils.tensor_to_numpy_safe(purified_importance)
        
        np.testing.assert_allclose(
            original_importance,
            purified_importance_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Feature importance differs between original and purified implementations"
        )
        
        logger.info(f"PCA blending: Original: {original_time:.4f}s, Purified: {purified_time:.4f}s")
        logger.info(f"Speedup: {original_time / purified_time:.2f}x")

    def test_process_batch(self, original_processor, purified_processor):
        """Test that batch processing produces equivalent results."""
        # Create sample data for batch processing
        data = np.random.randn(100, 5)
        
        # Process batch with original implementation
        start_time = time.time()
        original_result = original_processor.process_batch(data)
        original_time = time.time() - start_time
        
        # Reset PCA models to ensure fair comparison
        purified_processor.pca_models = {}
        
        # Process batch with purified implementation
        start_time = time.time()
        purified_result = purified_processor.process_batch(
            backend_utils.convert_to_tensor_safe(data)
        )
        purified_time = time.time() - start_time
        
        # Check that the results have the same strides
        assert set(original_result.keys()) == set(purified_result.keys())
        
        # Check that the results have the same shapes
        for stride in original_result.keys():
            original_shape = original_result[stride].shape
            purified_shape = backend_utils.tensor_to_numpy_safe(purified_result[stride]).shape
            assert original_shape == purified_shape
        
        logger.info(f"Batch processing: Original: {original_time:.4f}s, Purified: {purified_time:.4f}s")
        logger.info(f"Speedup: {original_time / purified_time:.2f}x")

    def test_end_to_end(self, original_processor, purified_processor):
        """Test end-to-end processing with both implementations."""
        # Create a simple data generator
        def data_generator(batch_size=20, num_batches=5):
            for _ in range(num_batches):
                yield np.random.randn(batch_size, 5)
        
        # Process data with original implementation
        start_time = time.time()
        original_result = original_processor.process_large_dataset(
            data_generator(), maintain_state=True
        )
        original_time = time.time() - start_time
        
        # Reset PCA models and state buffer to ensure fair comparison
        purified_processor.pca_models = {}
        purified_processor.state_buffer = None
        
        # Reset random seed
        np.random.seed(42)
        
        # Process data with purified implementation
        start_time = time.time()
        purified_result = purified_processor.process_large_dataset(
            data_generator(), maintain_state=True
        )
        purified_time = time.time() - start_time
        
        # Check that the results have the same strides
        assert set(original_result.keys()) == set(purified_result.keys())
        
        # Check that the results have the same shapes
        for stride in original_result.keys():
            original_shape = original_result[stride].shape
            purified_shape = backend_utils.tensor_to_numpy_safe(purified_result[stride]).shape
            assert original_shape == purified_shape
        
        logger.info(f"End-to-end processing: Original: {original_time:.4f}s, Purified: {purified_time:.4f}s")
        logger.info(f"Speedup: {original_time / purified_time:.2f}x")