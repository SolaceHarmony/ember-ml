"""
NumPy implementation of I/O operations.

This module provides NumPy implementations of the ember_ml I/O operations interface.
"""

import os
import numpy as np
from typing import Any, Optional

def save(filepath: str, obj: Any, allow_pickle: bool = True) -> None:
    """
    Save a tensor or dictionary of tensors to a file.
    
    Args:
        filepath: Path to save the object to
        obj: Tensor or dictionary of tensors to save
        allow_pickle: Whether to allow saving objects that can't be saved directly
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to file using NumPy
    np.save(filepath, obj, allow_pickle=allow_pickle)

def load(filepath: str, allow_pickle: bool = True) -> Any:
    """
    Load a tensor or dictionary of tensors from a file.
    
    Args:
        filepath: Path to load the object from
        allow_pickle: Whether to allow loading objects that can't be loaded directly
        
    Returns:
        Loaded tensor or dictionary of tensors
    """
    # Load from file using NumPy
    return np.load(filepath, allow_pickle=allow_pickle)

class NumPyIOOps:
    """NumPy implementation of I/O operations."""
    
    def save(self, filepath: str, obj: Any, allow_pickle: bool = True) -> None:
        """
        Save a tensor or dictionary of tensors to a file.
        
        Args:
            filepath: Path to save the object to
            obj: Tensor or dictionary of tensors to save
            allow_pickle: Whether to allow saving objects that can't be saved directly
            
        Returns:
            None
        """
        save(filepath, obj, allow_pickle)
    
    def load(self, filepath: str, allow_pickle: bool = True) -> Any:
        """
        Load a tensor or dictionary of tensors from a file.
        
        Args:
            filepath: Path to load the object from
            allow_pickle: Whether to allow loading objects that can't be loaded directly
            
        Returns:
            Loaded tensor or dictionary of tensors
        """
        return load(filepath, allow_pickle)