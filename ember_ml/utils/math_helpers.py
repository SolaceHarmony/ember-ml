"""
Math helper functions for the emberharmony library.

This module provides math helper functions for the emberharmony library.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
import math

def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the sigmoid function.
    
    Args:
        x: Input value or array
        
    Returns:
        Sigmoid of the input
    """
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the hyperbolic tangent function.
    
    Args:
        x: Input value or array
        
    Returns:
        Tanh of the input
    """
    return np.tanh(x)

def relu(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the ReLU function.
    
    Args:
        x: Input value or array
        
    Returns:
        ReLU of the input
    """
    return np.maximum(0, x)

def leaky_relu(x: Union[float, np.ndarray], alpha: float = 0.01) -> Union[float, np.ndarray]:
    """
    Compute the leaky ReLU function.
    
    Args:
        x: Input value or array
        alpha: Slope for negative values
        
    Returns:
        Leaky ReLU of the input
    """
    return np.maximum(alpha * x, x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the softmax function.
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax of the input
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalize an array to have unit norm.
    
    Args:
        x: Input array
        axis: Axis along which to normalize
        
    Returns:
        Normalized array
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-8)  # Add small epsilon to avoid division by zero

def standardize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Standardize an array to have zero mean and unit variance.
    
    Args:
        x: Input array
        axis: Axis along which to standardize
        
    Returns:
        Standardized array
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Cosine similarity
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)

def exponential_decay(initial_value: float, decay_rate: float, time_step: float) -> float:
    """
    Compute exponential decay.
    
    Args:
        initial_value: Initial value
        decay_rate: Decay rate
        time_step: Time step
        
    Returns:
        Decayed value
    """
    return initial_value * np.exp(-decay_rate * time_step)

def gaussian(x: Union[float, np.ndarray], mu: float = 0.0, sigma: float = 1.0) -> Union[float, np.ndarray]:
    """
    Compute the Gaussian function.
    
    Args:
        x: Input value or array
        mu: Mean
        sigma: Standard deviation
        
    Returns:
        Gaussian of the input
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))