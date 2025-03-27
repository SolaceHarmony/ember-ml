"""
Feature processing utilities for BigQuery data in Ember ML.

This module provides functions for processing numeric, categorical, and datetime
features from BigQuery data.
"""

import logging
from typing import List, Optional, Tuple, Any, Dict

import pandas as pd

from ember_ml.nn import tensor
from ember_ml import ops

logger = logging.getLogger(__name__)


def process_numeric_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    handle_missing: bool = True,
    handle_outliers: bool = True,
    normalize: bool = True,
    device: Optional[str] = None
) -> Tuple[Optional[Any], List[str]]:
    """
    Process numeric features from a DataFrame.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names to process
        handle_missing: Whether to handle missing values
        handle_outliers: Whether to handle outliers
        normalize: Whether to normalize features
        device: Device to place tensors on
        
    Returns:
        Tuple of (numeric_tensor, feature_names)
    """
    if not numeric_columns:
        return None, []
    
    # Extract numeric data
    numeric_data = df[numeric_columns].copy()
    
    # Convert to tensor
    numeric_tensor = tensor.convert_to_tensor(
        numeric_data.values,
        device=device
    )
    
    # Handle missing values
    if handle_missing:
        numeric_tensor = handle_missing_values(numeric_tensor)
    
    # Handle outliers
    if handle_outliers:
        numeric_tensor = handle_outliers(numeric_tensor)
    
    # Normalize features
    if normalize:
        numeric_tensor = normalize_robust(numeric_tensor)
    
    return numeric_tensor, numeric_columns


def process_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    handle_missing: bool = True,
    device: Optional[str] = None
) -> Tuple[Optional[Any], List[str]]:
    """
    Process categorical features from a DataFrame.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical column names to process
        handle_missing: Whether to handle missing values
        device: Device to place tensors on
        
    Returns:
        Tuple of (categorical_tensor, feature_names)
    """
    if not categorical_columns:
        return None, []
    
    all_cat_tensors = []
    all_cat_names = []
    
    for col in categorical_columns:
        # Get unique values
        unique_values = df[col].dropna().unique()
        
        # Skip columns with too many unique values (use hash encoding instead)
        if len(unique_values) > 100:
            tensor_data, feature_names = hash_encode(
                df, col, n_components=16, device=device
            )
        else:
            # One-hot encode
            tensor_data, feature_names = one_hot_encode(
                df, col, unique_values, device=device
            )
        
        all_cat_tensors.append(tensor_data)
        all_cat_names.extend(feature_names)
    
    # Concatenate all categorical tensors
    categorical_tensor = ops.concatenate(all_cat_tensors, axis=1)
    
    return categorical_tensor, all_cat_names


def process_datetime_features(
    df: pd.DataFrame,
    datetime_columns: List[str],
    device: Optional[str] = None
) -> Tuple[Optional[Any], List[str]]:
    """
    Process datetime features from a DataFrame.
    
    Args:
        df: Input DataFrame
        datetime_columns: List of datetime column names to process
        device: Device to place tensors on
        
    Returns:
        Tuple of (datetime_tensor, feature_names)
    """
    if not datetime_columns:
        return None, []
    
    # Extract datetime components
    datetime_features = []
    feature_names = []
    
    for col in datetime_columns:
        # Convert to pandas datetime
        datetime_series = pd.to_datetime(df[col], errors='coerce')
        
        # Extract components
        components = {
            f"{col}_year": datetime_series.dt.year,
            f"{col}_month": datetime_series.dt.month,
            f"{col}_day": datetime_series.dt.day,
            f"{col}_dayofweek": datetime_series.dt.dayofweek,
            f"{col}_hour": datetime_series.dt.hour if hasattr(datetime_series.dt, 'hour') else None,
            f"{col}_minute": datetime_series.dt.minute if hasattr(datetime_series.dt, 'minute') else None
        }
        
        # Filter out None components and convert to tensors
        for name, component in components.items():
            if component is not None:
                # Convert to tensor and normalize
                component_tensor = tensor.convert_to_tensor(
                    component.fillna(0).values,
                    device=device
                )
                
                # Reshape to column vector
                component_tensor = ops.reshape(
                    component_tensor, 
                    (tensor.shape(component_tensor)[0], 1)
                )
                
                # Normalize based on the component type
                if 'year' in name:
                    # Normalize year to [0, 1] range assuming years between 1900-2100
                    component_tensor = ops.divide(
                        ops.subtract(
                            component_tensor,
                            tensor.convert_to_tensor(1900, device=device)
                        ),
                        tensor.convert_to_tensor(200, device=device)
                    )
                elif 'month' in name:
                    # Normalize month to [0, 1] range
                    component_tensor = ops.divide(
                        ops.subtract(
                            component_tensor,
                            tensor.convert_to_tensor(1, device=device)
                        ),
                        tensor.convert_to_tensor(12, device=device)
                    )
                elif 'day' in name:
                    # Normalize day to [0, 1] range
                    component_tensor = ops.divide(
                        ops.subtract(
                            component_tensor,
                            tensor.convert_to_tensor(1, device=device)
                        ),
                        tensor.convert_to_tensor(31, device=device)
                    )
                elif 'dayofweek' in name:
                    # Normalize day of week to [0, 1] range
                    component_tensor = ops.divide(
                        component_tensor,
                        tensor.convert_to_tensor(6, device=device)
                    )
                elif 'hour' in name:
                    # Normalize hour to [0, 1] range
                    component_tensor = ops.divide(
                        component_tensor,
                        tensor.convert_to_tensor(23, device=device)
                    )
                elif 'minute' in name:
                    # Normalize minute to [0, 1] range
                    component_tensor = ops.divide(
                        component_tensor,
                        tensor.convert_to_tensor(59, device=device)
                    )
                
                datetime_features.append(component_tensor)
                feature_names.append(name)
    
    if not datetime_features:
        return None, []
    
    # Concatenate all datetime features
    datetime_tensor = ops.concatenate(datetime_features, axis=1)
    
    return datetime_tensor, feature_names


def handle_missing_values(data: Any) -> Any:
    """
    Handle missing values in numeric data.
    
    Args:
        data: Input tensor
        
    Returns:
        Tensor with missing values handled
    """
    # Check for NaN values
    nan_mask = ops.isnan(data)
    
    # If no NaNs, return the original data
    if not ops.any(nan_mask):
        return data
    
    # Calculate median for each feature
    # First, replace NaNs with zeros for calculation purposes
    data_no_nan = ops.where(nan_mask, ops.zeros_like(data), data)
    
    # Calculate median for each feature (column)
    medians = []
    for i in range(tensor.shape(data)[1]):
        col_data = data_no_nan[:, i]
        # Sort the data
        sorted_data = ops.sort(col_data)
        # Get median
        n = tensor.shape(sorted_data)[0]
        if n % 2 == 0:
            median = ops.divide(
                ops.add(sorted_data[n // 2 - 1], sorted_data[n // 2]),
                tensor.convert_to_tensor(2.0)
            )
        else:
            median = sorted_data[n // 2]
        medians.append(median)
    
    # Convert to tensor
    medians_tensor = ops.stack(medians)
    
    # Reshape medians to match data shape for broadcasting
    medians_tensor = ops.reshape(medians_tensor, (1, -1))
    
    # Replace NaNs with medians
    return ops.where(nan_mask, ops.broadcast_to(medians_tensor, tensor.shape(data)), data)


def handle_outliers(data: Any) -> Any:
    """
    Handle outliers in numeric data using IQR method.
    
    Args:
        data: Input tensor
        
    Returns:
        Tensor with outliers handled
    """
    # Calculate quartiles for each feature
    q1_values = []
    q3_values = []
    
    for i in range(tensor.shape(data)[1]):
        col_data = data[:, i]
        # Sort the data
        sorted_data = ops.sort(col_data)
        n = tensor.shape(sorted_data)[0]
        
        # Calculate quartile indices
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        
        # Get quartile values
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        
        q1_values.append(q1)
        q3_values.append(q3)
    
    # Convert to tensors
    q1_tensor = ops.stack(q1_values)
    q3_tensor = ops.stack(q3_values)
    
    # Calculate IQR
    iqr_tensor = ops.subtract(q3_tensor, q1_tensor)
    
    # Calculate bounds
    lower_bound = ops.subtract(q1_tensor, ops.multiply(
        iqr_tensor, tensor.convert_to_tensor(1.5)
    ))
    upper_bound = ops.add(q3_tensor, ops.multiply(
        iqr_tensor, tensor.convert_to_tensor(1.5)
    ))
    
    # Reshape bounds to match data shape for broadcasting
    lower_bound = ops.reshape(lower_bound, (1, -1))
    upper_bound = ops.reshape(upper_bound, (1, -1))
    
    # Clip values to bounds
    return ops.clip(data, lower_bound, upper_bound)


def normalize_robust(data: Any) -> Any:
    """
    Apply robust normalization to numeric data using quantiles.
    
    Args:
        data: Input tensor
        
    Returns:
        Normalized tensor
    """
    # Calculate min and max for each feature using 5th and 95th percentiles
    min_values = []
    max_values = []
    
    for i in range(tensor.shape(data)[1]):
        col_data = data[:, i]
        # Sort the data
        sorted_data = ops.sort(col_data)
        n = tensor.shape(sorted_data)[0]
        
        # Calculate percentile indices
        p05_idx = max(0, int(0.05 * n))
        p95_idx = min(n - 1, int(0.95 * n))
        
        # Get percentile values
        p05 = sorted_data[p05_idx]
        p95 = sorted_data[p95_idx]
        
        min_values.append(p05)
        max_values.append(p95)
    
    # Convert to tensors
    min_tensor = ops.stack(min_values)
    max_tensor = ops.stack(max_values)
    
    # Reshape min and max to match data shape for broadcasting
    min_tensor = ops.reshape(min_tensor, (1, -1))
    max_tensor = ops.reshape(max_tensor, (1, -1))
    
    # Calculate range
    range_tensor = ops.subtract(max_tensor, min_tensor)
    
    # Avoid division by zero
    epsilon = tensor.convert_to_tensor(1e-8, device=ops.get_device(data))
    range_tensor = ops.maximum(range_tensor, epsilon)
    
    # Normalize data
    normalized_data = ops.divide(
        ops.subtract(data, min_tensor),
        range_tensor
    )
    
    # Clip to ensure values are in [0, 1]
    return ops.clip(normalized_data, 0, 1)