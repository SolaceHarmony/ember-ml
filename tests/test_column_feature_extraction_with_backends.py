"""
Tests for the column-based feature extraction module with different backends.

This module contains tests for the ColumnFeatureExtractor, ColumnPCAFeatureExtractor,
and TemporalColumnFeatureExtractor classes, ensuring they work with all backends.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ember_ml.features.column_feature_extraction import (
    ColumnFeatureExtractor,
    ColumnPCAFeatureExtractor,
    TemporalColumnFeatureExtractor
)
from ember_ml.backend import get_backend, set_backend
from ember_ml import ops

# List of backends to test
BACKENDS = ['numpy']
try:
    import torch
    BACKENDS.append('torch')
except ImportError:
    pass

try:
    import mlx.core
    BACKENDS.append('mlx')
except ImportError:
    pass

@pytest.fixture(params=BACKENDS)
def backend(request):
    """Fixture to test with different backends."""
    prev_backend = get_backend()
    set_backend(request.param)
    ops.set_backend(request.param)
    yield request.param
    set_backend(prev_backend)
    ops.set_backend(prev_backend)

class TestColumnFeatureExtractor:
    """Tests for the ColumnFeatureExtractor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create a DataFrame with different column types
        return pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'categorical1': ['A', 'B', 'A', 'C', 'B'],
            'categorical2': [1, 2, 1, 3, 2],  # Numeric but should be treated as categorical
            'datetime1': pd.date_range('2023-01-01', periods=5),
            'text1': ['This is a test', 'Another test', 'Short', 'Longer text example', 'Final test']
        })
    
    def test_initialization(self, backend):
        """Test initialization of ColumnFeatureExtractor."""
        extractor = ColumnFeatureExtractor(
            numeric_strategy='standard',
            categorical_strategy='onehot',
            datetime_strategy='cyclical',
            text_strategy='basic',
            max_categories=100
        )
        
        assert extractor.numeric_strategy == 'standard'
        assert extractor.categorical_strategy == 'onehot'
        assert extractor.datetime_strategy == 'cyclical'
        assert extractor.text_strategy == 'basic'
        assert extractor.max_categories == 100
        assert extractor.column_processors == {}
        assert extractor.column_types == {}
        assert extractor.fitted is False
    
    def test_detect_column_types(self, sample_data, backend):
        """Test detection of column types."""
        extractor = ColumnFeatureExtractor()
        extractor._detect_column_types(sample_data)
        
        assert extractor.column_types['numeric1'] == 'numeric'
        assert extractor.column_types['numeric2'] == 'numeric'
        assert extractor.column_types['categorical1'] == 'categorical'
        assert extractor.column_types['categorical2'] == 'numeric'  # Detected as numeric
        assert extractor.column_types['datetime1'] == 'datetime'
        assert extractor.column_types['text1'] == 'categorical'  # Detected as categorical due to low cardinality
    
    def test_fit(self, sample_data, backend):
        """Test fitting the extractor to data."""
        extractor = ColumnFeatureExtractor()
        extractor.fit(sample_data)
        
        assert extractor.fitted is True
        assert len(extractor.column_processors) > 0
        
        # Check that processors were created for each column
        for column in sample_data.columns:
            assert column in extractor.column_types
    
    def test_transform_numeric(self, sample_data, backend):
        """Test transforming numeric columns."""
        extractor = ColumnFeatureExtractor(numeric_strategy='standard')
        extractor.fit(sample_data)
        
        transformed = extractor.transform(sample_data)
        
        # Check that numeric columns were transformed
        assert 'numeric1_scaled' in transformed.columns
        assert 'numeric2_scaled' in transformed.columns
        
        # Check that the values were standardized
        assert abs(transformed['numeric1_scaled'].mean()) < 1e-10
        assert abs(transformed['numeric2_scaled'].mean()) < 1e-10
    
    def test_transform_categorical(self, sample_data, backend):
        """Test transforming categorical columns."""
        extractor = ColumnFeatureExtractor(categorical_strategy='onehot')
        extractor.fit(sample_data)
        
        transformed = extractor.transform(sample_data)
        
        # Check that categorical columns were one-hot encoded
        assert 'categorical1_A' in transformed.columns
        assert 'categorical1_B' in transformed.columns
        assert 'categorical1_C' in transformed.columns
        
        # Check that the values are correct
        assert transformed.loc[0, 'categorical1_A'] == 1.0
        assert transformed.loc[0, 'categorical1_B'] == 0.0
        assert transformed.loc[0, 'categorical1_C'] == 0.0
    
    def test_transform_datetime(self, sample_data, backend):
        """Test transforming datetime columns."""
        extractor = ColumnFeatureExtractor(datetime_strategy='cyclical')
        extractor.fit(sample_data)
        
        transformed = extractor.transform(sample_data)
        
        # Check that datetime columns were transformed to cyclical features
        assert 'datetime1_sin_hour' in transformed.columns
        assert 'datetime1_cos_hour' in transformed.columns
        assert 'datetime1_sin_dayofweek' in transformed.columns
        assert 'datetime1_cos_dayofweek' in transformed.columns
        assert 'datetime1_sin_day' in transformed.columns
        assert 'datetime1_cos_day' in transformed.columns
        assert 'datetime1_sin_month' in transformed.columns
        assert 'datetime1_cos_month' in transformed.columns
    
    def test_transform_text(self, sample_data, backend):
        """Test transforming text columns."""
        # Force text1 to be detected as text
        sample_data['text1'] = [
            "This is a long text with many words that should be detected as text",
            "Another long text with different words and structure",
            "A third text sample with unique words and patterns",
            "Yet another text sample with more words",
            "The final text sample with even more unique words and patterns"
        ]
        
        extractor = ColumnFeatureExtractor(text_strategy='basic', max_categories=3)
        extractor.fit(sample_data)
        
        # Check that text1 was detected as text
        assert extractor.column_types['text1'] == 'text'
        
        transformed = extractor.transform(sample_data)
        
        # Check that text columns were transformed to basic features
        assert 'text1_length' in transformed.columns
        assert 'text1_word_count' in transformed.columns
        assert 'text1_char_per_word' in transformed.columns
        assert 'text1_uppercase_ratio' in transformed.columns
        assert 'text1_digit_ratio' in transformed.columns
        assert 'text1_special_ratio' in transformed.columns
    
    def test_fit_transform(self, sample_data, backend):
        """Test fit_transform method."""
        extractor = ColumnFeatureExtractor()
        transformed = extractor.fit_transform(sample_data)
        
        assert extractor.fitted is True
        assert len(transformed.columns) > len(sample_data.columns)
        assert transformed.shape[0] == sample_data.shape[0]


class TestColumnPCAFeatureExtractor:
    """Tests for the ColumnPCAFeatureExtractor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create a DataFrame with different column types
        return pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'numeric3': [0.1, 0.2, 0.3, 0.4, 0.5],
            'categorical1': ['A', 'B', 'A', 'C', 'B'],
            'categorical2': ['X', 'Y', 'X', 'Z', 'Y'],
            'datetime1': pd.date_range('2023-01-01', periods=5)
        })
    
    def test_initialization(self, backend):
        """Test initialization of ColumnPCAFeatureExtractor."""
        extractor = ColumnPCAFeatureExtractor(
            numeric_strategy='standard',
            categorical_strategy='onehot',
            datetime_strategy='cyclical',
            text_strategy='basic',
            max_categories=100,
            pca_components=2,
            pca_per_type=True
        )
        
        assert extractor.numeric_strategy == 'standard'
        assert extractor.categorical_strategy == 'onehot'
        assert extractor.datetime_strategy == 'cyclical'
        assert extractor.text_strategy == 'basic'
        assert extractor.max_categories == 100
        assert extractor.pca_components == 2
        assert extractor.pca_per_type is True
        assert extractor.pca_models == {}
    
    def test_fit(self, sample_data, backend):
        """Test fitting the extractor to data."""
        extractor = ColumnPCAFeatureExtractor(pca_components=2)
        extractor.fit(sample_data)
        
        assert extractor.fitted is True
        assert len(extractor.column_processors) > 0
        
        # Check that PCA models were created
        assert 'numeric' in extractor.pca_models
        
        # Check that the PCA model has the correct number of components
        assert extractor.pca_models['numeric'].n_components == 2
    
    def test_transform(self, sample_data, backend):
        """Test transforming data with PCA."""
        extractor = ColumnPCAFeatureExtractor(pca_components=2)
        extractor.fit(sample_data)
        
        transformed = extractor.transform(sample_data)
        
        # Check that PCA components were created
        assert 'pca_numeric_1' in transformed.columns
        assert 'pca_numeric_2' in transformed.columns
        
        # Check that the shape is correct
        assert transformed.shape[0] == sample_data.shape[0]
    
    def test_fit_transform(self, sample_data, backend):
        """Test fit_transform method."""
        extractor = ColumnPCAFeatureExtractor(pca_components=2)
        transformed = extractor.fit_transform(sample_data)
        
        assert extractor.fitted is True
        assert 'pca_numeric_1' in transformed.columns
        assert 'pca_numeric_2' in transformed.columns
        assert transformed.shape[0] == sample_data.shape[0]
    
    def test_pca_all_columns(self, sample_data, backend):
        """Test PCA on all columns together."""
        extractor = ColumnPCAFeatureExtractor(pca_components=2, pca_per_type=False)
        extractor.fit(sample_data)
        
        transformed = extractor.transform(sample_data)
        
        # Check that a single PCA model was created for all columns
        assert 'all' in extractor.pca_models
        
        # Check that PCA components were created
        assert 'pca_all_1' in transformed.columns
        assert 'pca_all_2' in transformed.columns


class TestTemporalColumnFeatureExtractor:
    """Tests for the TemporalColumnFeatureExtractor class."""
    
    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data for testing."""
        # Create a DataFrame with time series data
        dates = pd.date_range('2023-01-01', periods=20)
        return pd.DataFrame({
            'timestamp': dates,
            'value1': np.sin(np.linspace(0, 4 * np.pi, 20)),
            'value2': np.cos(np.linspace(0, 4 * np.pi, 20)),
            'category': ['A', 'B'] * 10
        })
    
    def test_initialization(self, backend):
        """Test initialization of TemporalColumnFeatureExtractor."""
        extractor = TemporalColumnFeatureExtractor(
            window_size=5,
            stride=1,
            numeric_strategy='standard',
            categorical_strategy='onehot',
            datetime_strategy='cyclical',
            text_strategy='basic',
            max_categories=100
        )
        
        assert extractor.window_size == 5
        assert extractor.stride == 1
        assert extractor.numeric_strategy == 'standard'
        assert extractor.categorical_strategy == 'onehot'
        assert extractor.datetime_strategy == 'cyclical'
        assert extractor.text_strategy == 'basic'
        assert extractor.max_categories == 100
        assert extractor.temporal_processors == {}
    
    def test_fit(self, sample_time_series, backend):
        """Test fitting the extractor to time series data."""
        extractor = TemporalColumnFeatureExtractor(window_size=3, stride=1)
        extractor.fit(sample_time_series, time_column='timestamp')
        
        assert extractor.fitted is True
        assert len(extractor.column_processors) > 0
        assert len(extractor.temporal_processors) > 0
        
        # Check that temporal processors were created for numeric columns
        assert 'value1_scaled' in extractor.temporal_processors
        assert 'value2_scaled' in extractor.temporal_processors
    
    def test_transform(self, sample_time_series, backend):
        """Test transforming time series data."""
        extractor = TemporalColumnFeatureExtractor(window_size=3, stride=1)
        extractor.fit(sample_time_series, time_column='timestamp')
        
        transformed = extractor.transform(sample_time_series, time_column='timestamp')
        
        # Check that temporal features were created
        assert 'value1_scaled_window_mean' in transformed.columns
        assert 'value1_scaled_window_std' in transformed.columns
        assert 'value1_scaled_window_min' in transformed.columns
        assert 'value1_scaled_window_max' in transformed.columns
        assert 'value1_scaled_window_slope' in transformed.columns
        
        assert 'value2_scaled_window_mean' in transformed.columns
        assert 'value2_scaled_window_std' in transformed.columns
        assert 'value2_scaled_window_min' in transformed.columns
        assert 'value2_scaled_window_max' in transformed.columns
        assert 'value2_scaled_window_slope' in transformed.columns
        
        # Check that the shape is correct (should have fewer rows due to windowing)
        assert transformed.shape[0] <= sample_time_series.shape[0]
    
    def test_fit_transform(self, sample_time_series, backend):
        """Test fit_transform method."""
        extractor = TemporalColumnFeatureExtractor(window_size=3, stride=1)
        transformed = extractor.fit_transform(sample_time_series, time_column='timestamp')
        
        assert extractor.fitted is True
        assert 'value1_scaled_window_mean' in transformed.columns
        assert transformed.shape[0] <= sample_time_series.shape[0]