"""
Visualization utilities for BigQuery data in Ember ML.

This module provides functions for visualizing features and processing steps
from BigQuery data.
"""

import time
import logging
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from ember_ml.nn import tensor

logger = logging.getLogger(__name__)


def create_sample_table(
    df: pd.DataFrame, 
    columns: List[str], 
    table_id: str, 
    title: str,
    max_rows: int = 5
) -> Dict[str, Any]:
    """
    Create a sample data table from DataFrame columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to include in the table
        table_id: Unique identifier for the table
        title: Title of the table
        max_rows: Maximum number of rows to include
        
    Returns:
        Dictionary containing the sample table information
    """
    # Sample rows from the DataFrame
    if len(df) > max_rows:
        sampled_df = df[columns].sample(max_rows)
    else:
        sampled_df = df[columns]
    
    # Create sample table
    sample_table = {
        'title': title,
        'data': sampled_df.to_dict('records')
    }
    
    return sample_table


def create_sample_table_from_tensor(
    data: Any,
    column_names: List[str],
    table_id: str,
    title: str,
    max_rows: int = 5
) -> Dict[str, Any]:
    """
    Create a sample data table from a tensor.
    
    Args:
        data: Input tensor
        column_names: Names for the columns
        table_id: Unique identifier for the table
        title: Title of the table
        max_rows: Maximum number of rows to include
        
    Returns:
        Dictionary containing the sample table information
    """
    # Convert tensor to numpy array for sampling
    data_np = tensor.to_numpy(data)
    
    # Sample rows
    if data_np.shape[0] > max_rows:
        # Use random indices to sample rows
        indices = list(range(data_np.shape[0]))
        import random
        random.shuffle(indices)
        sample_indices = indices[:max_rows]
        
        # Create sampled data
        sampled_data = []
        for idx in sample_indices:
            sampled_data.append(data_np[idx])
        
        # Convert to numpy array
        sampled_np = np.array(sampled_data)
    else:
        sampled_np = data_np
    
    # Create dictionary records
    records = []
    for i in range(sampled_np.shape[0]):
        record = {}
        for j, name in enumerate(column_names):
            if j < sampled_np.shape[1]:
                # Use float for values to avoid precision issues
                record[name] = float(sampled_np[i, j])
        records.append(record)
    
    # Create sample table
    sample_table = {
        'title': title,
        'data': records
    }
    
    return sample_table


def capture_frame(
    data: Any,
    step_name: str,
    feature_type: str
) -> Dict[str, Any]:
    """
    Capture a frame for animation.
    
    Args:
        data: Input tensor
        step_name: Name of the processing step
        feature_type: Type of features being processed
        
    Returns:
        Dictionary containing the frame information
    """
    # Convert tensor to numpy array for visualization
    data_np = tensor.to_numpy(data)
    
    # Create frame information
    frame = {
        'data': data_np,
        'step_name': step_name,
        'feature_type': feature_type,
        'timestamp': time.time()
    }
    
    return frame


def generate_processing_animation(frames: List[Dict[str, Any]]) -> Any:
    """
    Generate an animation from processing frames.
    
    This function can be implemented with various visualization libraries
    depending on requirements. For example, matplotlib for static visualizations
    or plotly for interactive visualizations.
    
    Args:
        frames: List of frame dictionaries
        
    Returns:
        Animation object
    """
    if not frames:
        logger.warning("No frames to generate animation")
        return None
    
    # Sort frames by timestamp
    sorted_frames = sorted(frames, key=lambda x: x['timestamp'])
    
    # This is a placeholder. The actual implementation would depend on
    # the specific visualization requirements and libraries used.
    logger.info(f"Generated animation with {len(sorted_frames)} frames")
    
    # Return animation metadata
    return {
        'frame_count': len(sorted_frames),
        'feature_types': list(set(frame['feature_type'] for frame in sorted_frames)),
        'steps': list(frame['step_name'] for frame in sorted_frames),
        'duration': sorted_frames[-1]['timestamp'] - sorted_frames[0]['timestamp']
    }