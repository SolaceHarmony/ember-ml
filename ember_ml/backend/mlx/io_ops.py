"""
MLX implementation of I/O operations.

This module provides MLX implementations of the ember_ml I/O operations interface.
"""

import os
import mlx.core as mx
from typing import Any, Optional
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.tensor import MLXTensor

Tensor = MLXTensor()


def save(filepath: str, obj: TensorLike, allow_pickle: Optional[bool] = None) -> None:
    """
    Save a tensor or dictionary of tensors to a file.
    
    Args:
        filepath: Path to save the object to
        obj: Tensor or dictionary of tensors to save
        allow_pickle: Not supported on MLX
        
    Returns:
        None
    """
    tensor = Tensor.convert_to_tensor(obj)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to file using MLX
    mx.save(filepath, tensor)

def load(filepath: str, allow_pickle: bool = True) -> Any:
    """
    Load a tensor or dictionary of tensors from a file.
    
    Args:
        filepath: Path to load the object from
        allow_pickle: Whether to allow loading objects that can't be loaded directly
        
    Returns:
        Loaded tensor or dictionary of tensors
    """
    # Load from file using MLX
    return mx.load(filepath)

class MLXIOOps:
    """MLX implementation of I/O operations."""
    
    def save(self, filepath: str, obj: TensorLike, allow_pickle: Optional[bool]) -> None:
        """
        Save a tensor or dictionary of tensors to a file.
        
        Args:
            filepath: Path to save the object to
            obj: Tensor or dictionary of tensors to save
            allow_pickle: Whether to allow saving objects that can't be saved directly
            
        Returns:
            None
        """
        save(filepath, obj)
    
    def load(self, filepath: str, allow_pickle: Optional[bool] = None) -> Any:
        """
        Load a tensor or dictionary of tensors from a file.
        
        Args:
            filepath: Path to load the object from
            allow_pickle: Whether to allow loading objects that can't be loaded directly
            
        Returns:
            Loaded tensor or dictionary of tensors
        """
        return load(filepath)