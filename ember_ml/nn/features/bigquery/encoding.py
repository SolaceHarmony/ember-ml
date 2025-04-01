"""
Encoding utilities for BigQuery data in Ember ML.

This module provides functions for encoding categorical features from BigQuery data.
"""

import logging
from typing import List, Optional, Tuple, Any

import pandas as pd

from ember_ml.nn import tensor

logger = logging.getLogger(__name__)


def hash_encode(
    df: pd.DataFrame,
    column: str,
    n_components: int = 16,
    seed: int = 42,
    device: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """
    Hash encode a high-cardinality categorical or identifier column.
    
    This method converts categorical values to a fixed-length vector representation
    using the hashing trick. This is useful for columns with many unique values
    where one-hot encoding would create too many dimensions.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        n_components: Number of hash components
        seed: Random seed
        device: Device to place tensors on
        
    Returns:
        Tuple of (encoded_tensor, feature_names)
    """
    # Get column data
    col_data = df[column]
    
    # Initialize encoded data
    n_samples = len(df)
    encoded_data = tensor.zeros((n_samples, n_components), device=device)
    
    # Generate feature names
    feature_names = [f"{column}_hash_{i}" for i in range(n_components)]
    
    # Set random seed
    tensor.set_seed(seed)
    
    # Encode each value
    for i, value in enumerate(col_data):
        if pd.isna(value):
            # Skip NaN values
            continue
            
        # Convert value to string
        str_value = str(value)
        
        # Generate hash value
        hash_val = hash(str_value + str(seed))
        
        # Use hash to generate pseudo-random values for each component
        for j in range(n_components):
            component_hash = hash(str_value + str(seed + j))
            # Scale to [0, 1]
            component_value = (component_hash % 10000) / 10000.0
            
            # Update the tensor at position [i, j]
            encoded_data = tensor.tensor_scatter_nd_update(
                encoded_data,
                tensor.reshape(tensor.convert_to_tensor([[i, j]], device=device), (1, 2)),
                tensor.reshape(tensor.convert_to_tensor([component_value], device=device), (1,))
            )
    
    # Reset seed
    tensor.set_seed(None)
    
    return encoded_data, feature_names


def one_hot_encode(
    df: pd.DataFrame,
    column: str,
    unique_values: Any,
    device: Optional[str] = None
) -> Tuple[Any, List[str]]:
    """
    One-hot encode a categorical column.
    
    This method converts categorical values to a one-hot encoded representation.
    Each unique value becomes a binary feature where 1 indicates the presence of
    the value and 0 indicates absence.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        unique_values: Unique values in the column
        device: Device to place tensors on
        
    Returns:
        Tuple of (encoded_tensor, feature_names)
    """
    # Get column data
    col_data = df[column]
    
    # Initialize encoded data
    n_samples = len(df)
    n_categories = len(unique_values)
    encoded_data = tensor.zeros((n_samples, n_categories), device=device)
    
    # Generate feature names
    feature_names = []
    value_to_index = {}
    
    for i, value in enumerate(unique_values):
        # Create safe feature name
        if isinstance(value, str):
            safe_value = value.replace(' ', '_').replace('-', '_').replace('/', '_')
        else:
            safe_value = str(value)
        
        feature_name = f"{column}_{safe_value}"
        feature_names.append(feature_name)
        value_to_index[value] = i
    
    # Encode each row
    for i, value in enumerate(col_data):
        if pd.isna(value):
            # Skip NaN values
            continue
            
        if value in value_to_index:
            idx = value_to_index[value]
            # Set the corresponding column to 1
            encoded_data = tensor.tensor_scatter_nd_update(
                encoded_data,
                tensor.reshape(tensor.convert_to_tensor([[i, idx]], device=device), (1, 2)),
                tensor.reshape(tensor.convert_to_tensor([1.0], device=device), (1,))
            )
    
    return encoded_data, feature_names