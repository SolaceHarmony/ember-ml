"""
Unit tests for the EnhancedTypeDetector class.

This module contains tests for the EnhancedTypeDetector class, ensuring
it correctly detects column types and generates sample tables.
"""

import unittest
import pandas as pd
from typing import Dict, List, Any

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.features.enhanced_type_detector import EnhancedTypeDetector


class TestEnhancedTypeDetector(unittest.TestCase):
    """Tests for the EnhancedTypeDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test DataFrame with various column types
        self.test_data = {
            'numeric_col': [1.0, 2.0, 3.0, 4.0, 5.0],
            'integer_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'high_cardinality_col': ['val1', 'val2', 'val3', 'val4', 'val5'],
            'datetime_col': pd.date_range('2025-01-01', periods=5),
            'id_col': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'boolean_col': [True, False, True, True, False],
            'text_col': [
                'This is a text value with multiple words.',
                'Another text value with words.',
                'More text for testing.',
                'Yet another text value.',
                'Final text value for the test.'
            ],
            'null_col': [None, None, 1.0, 2.0, None]
        }
        self.df = pd.DataFrame(self.test_data)
        
        # Create detector with visualization disabled for tests
        self.detector = EnhancedTypeDetector(
            visualization_enabled=False,
            sample_tables_enabled=True
        )
    
    def test_detect_column_types(self):
        """Test that column types are correctly detected."""
        # Detect column types
        column_types = self.detector.detect_column_types(self.df)
        
        # Check that all expected types are present
        self.assertIn('numeric', column_types)
        self.assertIn('categorical', column_types)
        self.assertIn('datetime', column_types)
        self.assertIn('identifier', column_types)
        
        # Check specific column categorizations
        self.assertIn('numeric_col', column_types['numeric'])
        self.assertIn('integer_col', column_types['numeric'])
        self.assertIn('categorical_col', column_types['categorical'])
        self.assertIn('datetime_col', column_types['datetime'])
        self.assertIn('id_col', column_types['identifier'])
        self.assertIn('boolean_col', column_types.get('boolean', []))
        self.assertIn('text_col', column_types.get('text', []))
        
        # Check that null_col is categorized as numeric due to non-null values
        self.assertIn('null_col', column_types['numeric'])
    
    def test_type_details_table(self):
        """Test that type details table is correctly generated."""
        # Detect column types first
        self.detector.detect_column_types(self.df)
        
        # Get the type details table
        type_details = self.detector.get_type_details_table()
        
        # Check that the table is not empty
        self.assertGreater(len(type_details), 0)
        
        # Check that all columns are present
        for col in self.df.columns:
            self.assertIn(col, type_details['Column'].values)
        
        # Check that the table has the expected columns
        expected_columns = ['Column', 'Detected Type', 'Cardinality', 
                           'Null Ratio', 'Recommended Strategy']
        for col in expected_columns:
            self.assertIn(col, type_details.columns)
    
    def test_sample_tables(self):
        """Test that sample tables are correctly generated."""
        # Detect column types with sample tables enabled
        self.detector.detect_column_types(self.df)
        
        # Check that sample tables were generated
        self.assertGreater(len(self.detector.sample_tables), 0)
        
        # Check structure of a sample table
        for table_id, table_data in self.detector.sample_tables.items():
            self.assertIn('title', table_data)
            self.assertIn('columns', table_data)
            self.assertIn('data', table_data)
            
            # Check that data is present for all columns
            for col in table_data['columns']:
                self.assertIn(col, table_data['data'])
    
    def test_forced_categorical(self):
        """Test that numeric columns can be forced to categorical."""
        # Create a new detector
        detector = EnhancedTypeDetector(
            visualization_enabled=False,
            sample_tables_enabled=False,
            cardinality_threshold=100  # High threshold to ensure numeric_col would be numeric
        )
        
        # Detect column types
        column_types = detector.detect_column_types(self.df)
        
        # Check that numeric_col is numeric
        self.assertIn('numeric_col', column_types['numeric'])
        
        # Move numeric_col to categorical
        from ember_ml.features.speedtest_event_processor import SpeedtestEventProcessor
        processor = SpeedtestEventProcessor(
            visualization_enabled=False,
            sample_tables_enabled=False
        )
        processor.column_types = column_types
        processor._move_columns_to_type(['numeric_col'], 'categorical')
        
        # Check that numeric_col is now categorical
        self.assertIn('numeric_col', processor.column_types['categorical'])
        self.assertNotIn('numeric_col', processor.column_types['numeric'])
    
    def test_null_ratio_calculation(self):
        """Test that null ratio is correctly calculated."""
        # Calculate null ratio for null_col
        null_ratio = self.detector._calculate_null_ratio(self.df['null_col'])
        
        # Expected ratio: 3/5 = 0.6
        self.assertAlmostEqual(null_ratio, 0.6, places=2)
    
    def test_cardinality_calculation(self):
        """Test that cardinality is correctly calculated."""
        # Calculate cardinality for categorical_col
        cardinality = self.detector._calculate_cardinality(self.df['categorical_col'])
        
        # Expected cardinality: 3 (A, B, C)
        self.assertEqual(cardinality, 3)
    
    def test_is_likely_identifier(self):
        """Test that identifier columns are correctly detected."""
        # Test with ID column
        self.assertTrue(self.detector._is_likely_identifier('id_col'))
        self.assertTrue(self.detector._is_likely_identifier('userId'))
        self.assertTrue(self.detector._is_likely_identifier('user_id'))
        
        # Test with non-ID column
        self.assertFalse(self.detector._is_likely_identifier('numeric_col'))
        self.assertFalse(self.detector._is_likely_identifier('text_value'))
    
    def test_is_likely_text(self):
        """Test that text columns are correctly detected."""
        # Test with text column
        self.assertTrue(self.detector._is_likely_text(self.df['text_col']))
        
        # Test with non-text column
        self.assertFalse(self.detector._is_likely_text(self.df['categorical_col']))


if __name__ == '__main__':
    unittest.main()