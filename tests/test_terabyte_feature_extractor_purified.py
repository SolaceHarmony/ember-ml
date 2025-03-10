"""
Tests for the purified version of TerabyteFeatureExtractor.

This module tests the backend-agnostic implementation of TerabyteFeatureExtractor
to ensure it works correctly with different backends.
"""

import pytest
import pandas as pd
import logging
import os
import sys

# Add parent directory to path to import ember_ml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_terabyte_feature_extractor')

# Import backend utilities and ops
from ember_ml.utils import backend_utils
from ember_ml import ops

# Import the TerabyteFeatureExtractor (now using the purified backend-agnostic implementation)
from ember_ml.features.terabyte_feature_extractor import TerabyteFeatureExtractor

# Import numpy only for testing assertions
import numpy as np

# Mock BigFrames for testing
class MockBigFramesDataFrame:
    """Mock BigFrames DataFrame for testing."""
    
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.columns = data.columns
        self.dtypes = data.dtypes
        self.index = data.index
    
    def __len__(self):
        return len(self.data)
    
    def iloc(self, *args, **kwargs):
        return self.data.iloc(*args, **kwargs)
    
    def to_pandas(self):
        return self.data
    
    def drop(self, *args, **kwargs):
        return MockBigFramesDataFrame(self.data.drop(*args, **kwargs))
    
    def set_index(self, *args, **kwargs):
        return MockBigFramesDataFrame(self.data.set_index(*args, **kwargs))


@pytest.fixture(scope="class")
def sample_data():
    """Create sample data for testing."""
    # Set random seed using backend-agnostic approach
    backend_utils.initialize_random_seed(42)
    
    # Generate random data using backend-agnostic operations
    numeric1 = backend_utils.tensor_to_numpy_safe(ops.random_normal(shape=(100,)))
    numeric2 = backend_utils.tensor_to_numpy_safe(ops.random_normal(shape=(100,)))
    
    # For categorical data, we need to use numpy directly for testing
    category1_idx = backend_utils.tensor_to_numpy_safe(
        ops.random_uniform(shape=(100,), minval=0, maxval=3).astype(ops.int32)
    )
    category2_idx = backend_utils.tensor_to_numpy_safe(
        ops.random_uniform(shape=(100,), minval=0, maxval=3).astype(ops.int32)
    )
    boolean1 = backend_utils.tensor_to_numpy_safe(
        ops.random_uniform(shape=(100,), minval=0, maxval=2).astype(ops.int32)
    ) > 0
    
    # Create sample DataFrame
    data = pd.DataFrame({
        'numeric1': numeric1,
        'numeric2': numeric2,
        'category1': [['A', 'B', 'C'][i] for i in category1_idx],
        'category2': [['X', 'Y', 'Z'][i] for i in category2_idx],
        'datetime1': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'boolean1': boolean1
    })
    
    # Create struct column
    data['struct1'] = [
        {'field1': i, 'field2': f'value{i}'} for i in range(100)
    ]
    
    return data


@pytest.fixture
def mock_bf_df(sample_data):
    """Create mock BigFrames DataFrame."""
    return MockBigFramesDataFrame(sample_data)


@pytest.fixture
def extractor_numpy():
    """Create extractor with NumPy backend."""
    extractor = TerabyteFeatureExtractor(
        project_id="test-project",
        location="US",
        preferred_backend="numpy"
    )
    
    # Store original method
    extractor._original_process_bigquery_in_chunks = extractor.process_bigquery_in_chunks
    
    return extractor


@pytest.fixture
def patched_extractor(extractor_numpy, sample_data):
    """Patch extractor methods that require BigFrames."""
    
    # Define mock implementation
    def _mock_process_bigquery_in_chunks(*args, **kwargs):
        """Mock implementation of process_bigquery_in_chunks."""
        # Return the sample data
        if 'processing_fn' in kwargs and kwargs['processing_fn'] is not None:
            return kwargs['processing_fn'](sample_data)
        return sample_data
    
    # Patch the method
    extractor_numpy.process_bigquery_in_chunks = _mock_process_bigquery_in_chunks
    
    yield extractor_numpy
    
    # Restore original method
    extractor_numpy.process_bigquery_in_chunks = extractor_numpy._original_process_bigquery_in_chunks


class TestTerabyteFeatureExtractorPurified:
    """Test cases for the purified TerabyteFeatureExtractor."""
    
    def test_backend_selection(self, extractor_numpy):
        """Test that backend selection works correctly."""
        # Test NumPy backend
        assert extractor_numpy.backend == 'numpy'
        
        # Test other backends if available
        try:
            extractor_mlx = TerabyteFeatureExtractor(preferred_backend="mlx")
            assert extractor_mlx.backend == 'mlx'
        except Exception as e:
            logger.info(f"MLX backend not available: {e}")
        
        try:
            extractor_torch = TerabyteFeatureExtractor(preferred_backend="torch")
            assert extractor_torch.backend == 'torch'
        except Exception as e:
            logger.info(f"PyTorch backend not available: {e}")
    
    def test_random_generation(self):
        """Test that random generation is backend-agnostic."""
        # Generate random values
        random_values = backend_utils.random_uniform(100)
        
        # Convert to numpy for testing
        random_np = backend_utils.tensor_to_numpy_safe(random_values)
        
        # Check properties
        assert len(random_np) == 100
        assert np.all(random_np >= 0.0)
        assert np.all(random_np <= 1.0)
    
    def test_sin_cos_transform(self):
        """Test that sine/cosine transformations are backend-agnostic."""
        # Create test values using backend-agnostic operations
        values_tensor = ops.linspace(0, 1, 100)
        values = backend_utils.tensor_to_numpy_safe(values_tensor)
        
        # Apply transformations
        sin_values, cos_values = backend_utils.sin_cos_transform(values_tensor)
        
        # Convert to numpy for testing
        sin_np = backend_utils.tensor_to_numpy_safe(sin_values)
        cos_np = backend_utils.tensor_to_numpy_safe(cos_values)
        
        # Check against expected values (using numpy only for assertions)
        expected_sin = np.sin(2 * np.pi * values)
        expected_cos = np.cos(2 * np.pi * values)
        
        np.testing.assert_allclose(sin_np, expected_sin, rtol=1e-5)
        np.testing.assert_allclose(cos_np, expected_cos, rtol=1e-5)
    
    def test_datetime_features(self, patched_extractor):
        """Test creation of datetime features."""
        # Create sample DataFrame with datetime column
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D')
        })
        
        # Create datetime features
        result_df = patched_extractor._create_datetime_features(df, 'date')
        
        # Check that expected columns were created
        expected_columns = [
            'date',
            'date_sin_hour', 'date_cos_hour',
            'date_sin_dayofweek', 'date_cos_dayofweek',
            'date_sin_day', 'date_cos_day',
            'date_sin_month', 'date_cos_month'
        ]
        
        for col in expected_columns:
            assert col in result_df.columns
        
        # Check values for first row
        first_date = df['date'].iloc[0]  # 2023-01-01
        
        # Hour should be 0, so sin(0) = 0, cos(0) = 1
        assert result_df['date_sin_hour'].iloc[0] == pytest.approx(0.0, abs=1e-5)
        assert result_df['date_cos_hour'].iloc[0] == pytest.approx(1.0, abs=1e-5)
        
        # Day of week for 2023-01-01 is Sunday (6), so sin(2π*6/6) = 0, cos(2π*6/6) = 1
        assert result_df['date_sin_dayofweek'].iloc[0] == pytest.approx(0.0, abs=1e-5)
        assert result_df['date_cos_dayofweek'].iloc[0] == pytest.approx(1.0, abs=1e-5)
        
        # Day of month is 1, so sin(2π*0/30) = 0, cos(2π*0/30) = 1
        assert result_df['date_sin_day'].iloc[0] == pytest.approx(0.0, abs=1e-5)
        assert result_df['date_cos_day'].iloc[0] == pytest.approx(1.0, abs=1e-5)
        
        # Month is 1, so sin(2π*0/11) = 0, cos(2π*0/11) = 1
        assert result_df['date_sin_month'].iloc[0] == pytest.approx(0.0, abs=1e-5)
        assert result_df['date_cos_month'].iloc[0] == pytest.approx(1.0, abs=1e-5)
    
    def test_split_data(self, patched_extractor):
        """Test data splitting with backend-agnostic random generation."""
        # Create sample DataFrame using backend-agnostic operations
        feature1 = backend_utils.tensor_to_numpy_safe(ops.random_normal(shape=(100,)))
        feature2 = backend_utils.tensor_to_numpy_safe(ops.random_normal(shape=(100,)))
        target = backend_utils.tensor_to_numpy_safe(ops.random_normal(shape=(100,)))
        
        df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'target': target
        })
        
        # Split data
        train_df, val_df, test_df = patched_extractor._split_data(df, None)
        
        # Check split sizes
        assert len(train_df) > 70  # ~80% for training
        assert len(val_df) > 5     # ~10% for validation
        assert len(test_df) > 5    # ~10% for testing
        
        # Check total size
        assert len(train_df) + len(val_df) + len(test_df) == 100
    
    def test_prepare_data(self, patched_extractor):
        """Test the full data preparation pipeline."""
        # Prepare data
        result = patched_extractor.prepare_data(
            table_id="test-dataset.test-table",
            target_column="numeric1",
            force_categorical_columns=["category1", "category2"],
            limit=100
        )
        
        # Check that result is not None
        assert result is not None
        
        # Unpack result
        train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer = result
        
        # Check that DataFrames are not empty
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        
        # Check that features were extracted
        assert len(train_features) > 0
        assert train_features == val_features
        assert train_features == test_features
        
        # Check that scaler and imputer were created
        assert scaler is not None
        assert imputer is not None