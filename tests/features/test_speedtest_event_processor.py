"""
Unit tests for the SpeedtestEventProcessor class.

This module contains tests for the SpeedtestEventProcessor class, ensuring
it correctly processes speedtest event data from BigQuery.
"""

import unittest
import pandas as pd
import numpy as np  # Used only for test data creation, not in the actual implementation
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.features.speedtest_event_processor import SpeedtestEventProcessor


class TestSpeedtestEventProcessor(unittest.TestCase):
    """Tests for the SpeedtestEventProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create processor with visualization and sample tables disabled for tests
        self.processor = SpeedtestEventProcessor(
            visualization_enabled=False,
            sample_tables_enabled=False
        )
        
        # Create a mock DataFrame that mimics the structure of the speedtest event table
        self.test_data = {
            'deviceId': ['dev-001', 'dev-002', 'dev-003', 'dev-004', 'dev-005'],
            'eventType': ['SPEEDTEST', 'SPEEDTEST', 'ERROR', 'SPEEDTEST', 'SPEEDTEST'],
            'downloadLatency': [35.2, 42.8, None, 28.3, 58.4],
            'uploadLatency': [48.7, 53.1, None, 39.5, 67.2],
            'downloadSpeed': [56.8, 48.2, None, 72.1, 32.6],
            'uploadSpeed': [12.3, 10.7, None, 15.8, 8.4],
            'testTimestamp': pd.date_range('2025-03-24 09:15:22', periods=5, freq='H'),
            'deviceType': ['modem-XJ5', 'modem-XJ5', 'modem-XJ6', 'modem-XJ5', 'modem-XJ6']
        }
        self.mock_df = pd.DataFrame(self.test_data)
    
    @patch('ember_ml.features.speedtest_event_processor.SpeedtestEventProcessor._load_bigquery_data')
    def test_process(self, mock_load_data):
        """Test the main processing pipeline."""
        # Mock the _load_bigquery_data method to return our test DataFrame
        mock_load_data.return_value = self.mock_df
        
        # Process the data
        result = self.processor.process(
            table_id="TEST1.ctl_modem_speedtest_event",
            limit=5,
            force_categorical_columns=['eventType', 'deviceType'],
            force_identifier_columns=['deviceId']
        )
        
        # Check that processing was successful
        self.assertIsNotNone(result)
        self.assertIsNotNone(result['features'])
        
        # Check column types
        column_types = result['column_types']
        self.assertIn('categorical', column_types)
        self.assertIn('numeric', column_types)
        self.assertIn('datetime', column_types)
        self.assertIn('identifier', column_types)
        
        # Check specific columns were correctly categorized
        self.assertIn('eventType', column_types['categorical'])
        self.assertIn('deviceType', column_types['categorical'])
        self.assertIn('deviceId', column_types['identifier'])
        self.assertIn('downloadLatency', column_types['numeric'])
        self.assertIn('testTimestamp', column_types['datetime'])
        
        # Check that feature tensors were created
        feature_tensors = result['feature_tensors']
        self.assertIn('numeric', feature_tensors)
        self.assertIn('categorical', feature_tensors)
        self.assertIn('datetime', feature_tensors)
        self.assertIn('identifier', feature_tensors)
        
        # Check combined features shape
        combined_features = result['features']
        shape = ops.shape(combined_features)
        self.assertEqual(shape[0], 5)  # 5 rows
        self.assertGreater(shape[1], 0)  # At least some features
    
    def test_move_columns_to_type(self):
        """Test that columns can be moved between types."""
        # Set up initial column types
        self.processor.column_types = {
            'numeric': ['downloadLatency', 'uploadLatency', 'downloadSpeed', 'uploadSpeed'],
            'categorical': ['eventType', 'deviceType'],
            'datetime': ['testTimestamp'],
            'identifier': ['deviceId']
        }
        
        # Move a numeric column to categorical
        self.processor._move_columns_to_type(['downloadLatency'], 'categorical')
        
        # Check that the column was moved
        self.assertIn('downloadLatency', self.processor.column_types['categorical'])
        self.assertNotIn('downloadLatency', self.processor.column_types['numeric'])
        
        # Move a categorical column to identifier
        self.processor._move_columns_to_type(['deviceType'], 'identifier')
        
        # Check that the column was moved
        self.assertIn('deviceType', self.processor.column_types['identifier'])
        self.assertNotIn('deviceType', self.processor.column_types['categorical'])
    
    def test_apply_forced_column_types(self):
        """Test that forced column types are correctly applied."""
        # Set up initial column types
        self.processor.column_types = {
            'numeric': ['downloadLatency', 'uploadLatency', 'downloadSpeed', 'uploadSpeed'],
            'categorical': ['eventType'],
            'datetime': ['testTimestamp'],
            'identifier': ['deviceId']
        }
        
        # Apply forced column types
        self.processor._apply_forced_column_types(
            force_categorical_columns=['downloadLatency', 'deviceType'],
            force_numeric_columns=['eventType'],
            force_datetime_columns=[],
            force_identifier_columns=[]
        )
        
        # Check that columns were moved
        self.assertIn('downloadLatency', self.processor.column_types['categorical'])
        self.assertNotIn('downloadLatency', self.processor.column_types['numeric'])
        
        self.assertIn('eventType', self.processor.column_types['numeric'])
        self.assertNotIn('eventType', self.processor.column_types['categorical'])
        
        # Check that deviceType was added to categorical
        self.assertIn('deviceType', self.processor.column_types['categorical'])
    
    def test_prepare_for_rbm(self):
        """Test that features are correctly prepared for RBM."""
        # Create a test tensor
        test_tensor = tensor.convert_to_tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.8, 0.7, 0.6],
            [0.4, 0.3, 0.2, 0.1],
            [0.5, 0.5, 0.5, 0.5]
        ])
        
        # Set the combined features
        self.processor.combined_features = test_tensor
        
        # Test binary preparation
        binary_features, binary_metadata = self.processor.prepare_for_rbm(
            binary_visible=True,
            binarization_threshold=0.5
        )
        
        # Check that binary features were created
        self.assertIsNotNone(binary_features)
        
        # Check that values are binary (0 or 1)
        binary_np = tensor.to_numpy(binary_features)
        unique_values = np.unique(binary_np)
        self.assertEqual(len(unique_values), 2)
        self.assertTrue(np.all(np.isin(unique_values, [0.0, 1.0])))
        
        # Test continuous preparation
        continuous_features, continuous_metadata = self.processor.prepare_for_rbm(
            binary_visible=False
        )
        
        # Check that continuous features were created
        self.assertIsNotNone(continuous_features)
        
        # Check that values are the same as the original
        self.assertTrue(ops.all(ops.equal(continuous_features, test_tensor)))
    
    def test_track_memory_usage(self):
        """Test that memory usage is correctly tracked."""
        # Track memory usage
        self.processor._track_memory_usage("Test Stage")
        
        # Check that memory data was recorded
        self.assertGreaterEqual(len(self.processor.memory_usage_data), 1)
        
        # Check structure of memory data
        last_entry = self.processor.memory_usage_data[-1]
        self.assertIn('stage', last_entry)
        self.assertIn('timestamp', last_entry)
        self.assertIn('memory_gb', last_entry)
        self.assertIn('percent_of_max', last_entry)
        
        # Check values
        self.assertEqual(last_entry['stage'], "Test Stage")
        self.assertGreaterEqual(last_entry['memory_gb'], 0.0)
        self.assertGreaterEqual(last_entry['percent_of_max'], 0.0)
    
    def test_empty_result(self):
        """Test that empty result is correctly created."""
        # Get empty result
        result = self.processor._empty_result()
        
        # Check structure
        self.assertIsNone(result['features'])
        self.assertEqual(result['feature_tensors'], {})
        self.assertEqual(result['column_types'], {})
        self.assertIn('error', result['metadata'])
        
        # Check that sample tables are empty
        self.assertEqual(result['sample_tables']['type_detector'], {})
        self.assertEqual(result['sample_tables']['feature_processor'], {})


if __name__ == '__main__':
    unittest.main()