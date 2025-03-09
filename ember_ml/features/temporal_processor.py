"""
Temporal Stride Processor

This module provides a class for processing data into multi-stride temporal representations.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.decomposition import PCA


class TemporalStrideProcessor:
    """
    Processes data into multi-stride temporal representations.
    
    This class creates sliding windows with different strides and applies
    PCA for dimensionality reduction, enabling multi-scale temporal analysis.
    """
    
    def __init__(self, window_size: int = 5, stride_perspectives: List[int] = None,
                 pca_components: Optional[int] = None):
        """
        Initialize the temporal stride processor.
        
        Args:
            window_size: Size of the sliding window
            stride_perspectives: List of stride lengths to use
            pca_components: Number of PCA components (if None, will be calculated)
        """
        self.window_size = window_size
        self.stride_perspectives = stride_perspectives or [1, 3, 5]
        self.pca_components = pca_components
        self.pca_models = {}  # Store PCA models for each stride
        
    def process_batch(self, data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Process data into multi-stride temporal representations.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Dictionary of stride perspectives with processed data
        """
        results = {}
        
        for stride in self.stride_perspectives:
            # Extract windows using stride length
            windows = self._create_strided_sequences(data, stride)
            
            if not windows:
                print(f"Warning: No windows created for stride {stride}")
                continue
                
            # Convert to array and apply PCA blending
            windows_array = np.array(windows)
            results[stride] = self._apply_pca_blend(windows_array, stride)
            
            print(f"Created {len(windows)} windows with stride {stride}, "
                  f"shape after PCA: {results[stride].shape}")
            
        return results
    
    def _create_strided_sequences(self, data: np.ndarray, stride: int) -> List[np.ndarray]:
        """
        Create sequences with the given stride.
        
        Args:
            data: Input data array
            stride: Stride length
            
        Returns:
            List of windowed sequences
        """
        num_samples = len(data)
        windows = []
        
        # Skip if data is too small for even one window
        if num_samples < self.window_size:
            print(f"Warning: Data length ({num_samples}) is smaller than window size ({self.window_size})")
            return windows
        
        for i in range(0, num_samples - self.window_size + 1, stride):
            windows.append(data[i:i+self.window_size])
            
        return windows
    
    def _apply_pca_blend(self, window_batch: np.ndarray, stride: int) -> np.ndarray:
        """
        Apply PCA-based temporal blending.
        
        Args:
            window_batch: Batch of windows (batch_size x window_size x features)
            stride: Stride length
            
        Returns:
            PCA-transformed data
        """
        batch_size, window_size, feature_dim = window_batch.shape
        
        # Reshape for PCA: [batch_size, window_size * feature_dim]
        flat_windows = window_batch.reshape(batch_size, -1)
        
        # Ensure PCA is fit
        if stride not in self.pca_models:
            # Calculate appropriate number of components
            if self.pca_components is None:
                # Use half the flattened dimension, but cap at 32 components
                n_components = min(flat_windows.shape[1] // 2, 32)
                # Ensure we don't try to extract more components than samples
                n_components = min(n_components, batch_size - 1)
            else:
                n_components = min(self.pca_components, batch_size - 1, flat_windows.shape[1])
                
            print(f"Fitting PCA for stride {stride} with {n_components} components")
            self.pca_models[stride] = PCA(n_components=n_components)
            self.pca_models[stride].fit(flat_windows)
            
        # Transform the data
        return self.pca_models[stride].transform(flat_windows)
    
    def get_explained_variance(self, stride: int) -> Optional[float]:
        """
        Get the explained variance ratio for a specific stride.
        
        Args:
            stride: Stride length
            
        Returns:
            Sum of explained variance ratios or None if PCA not fit
        """
        if stride in self.pca_models:
            return sum(self.pca_models[stride].explained_variance_ratio_)
        return None
    
    def get_feature_importance(self, stride: int) -> Optional[np.ndarray]:
        """
        Get feature importance for a specific stride.
        
        Args:
            stride: Stride length
            
        Returns:
            Array of feature importance scores or None if PCA not fit
        """
        if stride in self.pca_models:
            # Calculate feature importance as the sum of absolute component weights
            return np.abs(self.pca_models[stride].components_).sum(axis=0)
        return None