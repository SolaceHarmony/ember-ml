"""
Unit tests for the AnimatedFeatureProcessor class.

This module contains tests for the AnimatedFeatureProcessor class, ensuring
it correctly processes different types of features.
"""

import unittest
import pandas as pd
import numpy as np  # Used only for test data creation, not in the actual implementation
from typing import Dict, List, Any

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.features.animated_feature_processor import AnimatedFeatureProcessor


class TestAnimatedFeatureProcessor(unittest.TestCase):
    """Tests for the AnimatedFeatureProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test DataFrame with various column types
        self.test_data = {
            'numeric_col1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric_col2': [10.0, 20.0, np.nan, 40.0, 50.0],  # With NaN
            'categorical_col1': ['A', 'B', 'A', 'C', 'B'],
            'categorical_col2': ['X', 'Y', 'Z', 'X', 'Y'],
            'datetime_col': pd.date_range('2025-01-01', periods=5),
            'id_col': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005']
        }
        self.df = pd.DataFrame(self.test_data)
        
        # Create processor with visualization disabled for tests
        self.processor = AnimatedFeatureProcessor(
            visualization_enabled=False,
            sample_tables_enabled=True
        )
    
    def test_process_numeric_features(self):
        """Test that numeric features are correctly processed."""
        # Get numeric columns
        numeric_columns = ['numeric_col1', 'numeric_col2']
        
        # Process numeric features
        result = self.processor.process_numeric_features(self.df, numeric_columns)
        
        # Check that result is a tensor
        self.assertIsNotNone(result)
        
        # Check shape (5 rows, 2 columns)
        shape = ops.shape(result)
        self.assertEqual(shape[0], 5)
        self.assertEqual(shape[1], 2)
        
        # Check sample tables were generated
        self.assertIn('original_numeric', self.processor.sample_tables)
        self.assertIn('normalized_numeric', self.processor.sample_tables)
    
    def test_process_categorical_features(self):
        """Test that categorical features are correctly processed."""
        # Get categorical columns
        categorical_columns = ['categorical_col1', 'categorical_col2']
        
        # Process categorical features
        result = self.processor.process_categorical_features(self.df, categorical_columns)
        
        # Check that result is a tensor
        self.assertIsNotNone(result)
        
        # Check shape (5 rows, sum of unique values in each column)
        # categorical_col1 has 3 unique values, categorical_col2 has 3 unique values
        shape = ops.shape(result)
        self.assertEqual(shape[0], 5)
        self.assertEqual(shape[1], 6)  # 3 + 3 = 6
        
        # Check sample tables were generated
        self.assertIn('original_categorical', self.processor.sample_tables)
        self.assertIn('encoded_categorical', self.processor.sample_tables)
    
    def test_process_datetime_features(self):
        """Test that datetime features are correctly processed."""
        # Get datetime columns
        datetime_columns = ['datetime_col']
        
        # Process datetime features
        result = self.processor.process_datetime_features(self.df, datetime_columns)
        
        # Check that result is a tensor
        self.assertIsNotNone(result)
        
        # Check shape (5 rows, multiple features per datetime column)
        shape = ops.shape(result)
        self.assertEqual(shape[0], 5)
        
        # We expect at least 2 features per component (sin/cos) for each datetime unit
        # If we include year, month, day, hour, weekday, that's at least 5*2 = 10 features
        self.assertGreaterEqual(shape[1], 10)
        
        # Check sample tables were generated
        self.assertIn('original_datetime', self.processor.sample_tables)
        self.assertIn('encoded_datetime', self.processor.sample_tables)
    
    def test_process_identifier_features(self):
        """Test that identifier features are correctly processed."""
        # Get identifier columns
        identifier_columns = ['id_col']
        
        # Process identifier features
        result = self.processor.process_identifier_features(self.df, identifier_columns)
        
        # Check that result is a tensor
        self.assertIsNotNone(result)
        
        # Check shape (5 rows, n_components per identifier column)
        shape = ops.shape(result)
        self.assertEqual(shape[0], 5)
        self.assertEqual(shape[1], 16)  # Default n_components=16
        
        # Check sample tables were generated
        self.assertIn('original_identifier', self.processor.sample_tables)
        self.assertIn('encoded_identifier', self.processor.sample_tables)
    
    def test_handle_missing_values(self):
        """Test that missing values are correctly handled."""
        # Create tensor with missing values
        data = self._dataframe_to_tensor(self.df[['numeric_col1', 'numeric_col2']])
        
        # Handle missing values
        result = self.processor._handle_missing_values(data)
        
        # Check that result is a tensor
        self.assertIsNotNone(result)
        
        # Check shape (should be unchanged)
        self.assertEqual(ops.shape(result), ops.shape(data))
        
        # Check that there are no NaNs in the result
        nan_mask = ops.isnan(result)
        self.assertFalse(ops.any(nan_mask))
    
    def test_handle_outliers(self):
        """Test that outliers are correctly handled."""
        # Create tensor with outliers
        outlier_data = {
            'numeric_col1': [1.0, 2.0, 100.0, 4.0, 5.0],  # 100.0 is an outlier
            'numeric_col2': [10.0, 20.0, 30.0, 40.0, 500.0]  # 500.0 is an outlier
        }
        df_outliers = pd.DataFrame(outlier_data)
        data = self._dataframe_to_tensor(df_outliers)
        
        # Handle outliers
        result = self.processor._handle_outliers(data)
        
        # Check that result is a tensor
        self.assertIsNotNone(result)
        
        # Check shape (should be unchanged)
        self.assertEqual(ops.shape(result), ops.shape(data))
        
        # Convert to numpy for easier inspection
        result_np = tensor.to_numpy(result)
        data_np = tensor.to_numpy(data)
        
        # Check that outliers have been clipped
        self.assertLess(result_np[2, 0], data_np[2, 0])  # 100.0 should be reduced
        self.assertLess(result_np[4, 1], data_np[4, 1])  # 500.0 should be reduced
    
    def test_normalize_robust(self):
        """Test that data is correctly normalized."""
        # Create tensor with known range
        range_data = {
            'numeric_col1': [0.0, 25.0, 50.0, 75.0, 100.0],
            'numeric_col2': [0.0, 250.0, 500.0, 750.0, 1000.0]
        }
        df_range = pd.DataFrame(range_data)
        data = self._dataframe_to_tensor(df_range)
        
        # Normalize data
        result = self.processor._normalize_robust(data)
        
        # Check that result is a tensor
        self.assertIsNotNone(result)
        
        # Check shape (should be unchanged)
        self.assertEqual(ops.shape(result), ops.shape(data))
        
        # Convert to numpy for easier inspection
        result_np = tensor.to_numpy(result)
        
        # Check that values are in [0, 1] range
        self.assertGreaterEqual(result_np.min(), 0.0)
        self.assertLessEqual(result_np.max(), 1.0)
        
        # Check that the normalized values maintain relative ordering
        self.assertLess(result_np[0, 0], result_np[1, 0])
        self.assertLess(result_np[1, 0], result_np[2, 0])
    
    def test_one_hot_encode(self):
        """Test that one-hot encoding works correctly."""
        # Get a categorical column
        column = 'categorical_col1'
        unique_values = self.df[column].unique()
        
        # Encode the column
        encoded_data, feature_names = self.processor._one_hot_encode(
            self.df, column, unique_values
        )
        
        # Check that result is a tensor
        self.assertIsNotNone(encoded_data)
        
        # Check shape (5 rows, 3 unique values)
        shape = ops.shape(encoded_data)
        self.assertEqual(shape[0], 5)
        self.assertEqual(shape[1], 3)
        
        # Check feature names
        self.assertEqual(len(feature_names), 3)
        for name in feature_names:
            self.assertTrue(name.startswith(column))
    
    def test_hash_encode(self):
        """Test that hash encoding works correctly."""
        # Get an identifier column
        column = 'id_col'
        n_components = 8
        
        # Encode the column
        encoded_data, feature_names = self.processor._hash_encode(
            self.df, column, n_components=n_components
        )
        
        # Check that result is a tensor
        self.assertIsNotNone(encoded_data)
        
        # Check shape (5 rows, n_components)
        shape = ops.shape(encoded_data)
        self.assertEqual(shape[0], 5)
        self.assertEqual(shape[1], n_components)
        
        # Check feature names
        self.assertEqual(len(feature_names), n_components)
        for name in feature_names:
            self.assertTrue(name.startswith(column))
    
    def _dataframe_to_tensor(self, df):
        """Convert DataFrame to tensor for testing."""
        return self.processor._dataframe_to_tensor(df)


if __name__ == '__main__':
    unittest.main()