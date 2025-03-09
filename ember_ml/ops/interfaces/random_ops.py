"""
Random operations interface.

This module defines the abstract interface for random operations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional, Any, Tuple

# Type aliases
Shape = Union[int, Sequence[int]]
DType = Any  # Each backend will define its own dtype

class RandomOps(ABC):
    """Abstract interface for random operations."""
    
    @abstractmethod
    def random_normal(self, shape: Shape, mean: float = 0.0, stddev: float = 1.0, 
                     dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from a normal distribution.
        
        Args:
            shape: Shape of the tensor
            mean: Mean of the normal distribution
            stddev: Standard deviation of the normal distribution
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random normal values
        """
        pass
    
    @abstractmethod
    def random_uniform(self, shape: Shape, minval: float = 0.0, maxval: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from a uniform distribution.
        
        Args:
            shape: Shape of the tensor
            minval: Minimum value
            maxval: Maximum value
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random uniform values
        """
        pass
    
    @abstractmethod
    def random_binomial(self, shape: Shape, p: float = 0.5,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from a binomial distribution.
        
        Args:
            shape: Shape of the tensor
            p: Probability of success
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random binomial values
        """
        pass
    
    @abstractmethod
    def random_gamma(self, shape: Shape, alpha: float = 1.0, beta: float = 1.0,
                    dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from a gamma distribution.
        
        Args:
            shape: Shape of the tensor
            alpha: Shape parameter
            beta: Scale parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random gamma values
        """
        pass
    
    @abstractmethod
    def random_poisson(self, shape: Shape, lam: float = 1.0,
                      dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from a Poisson distribution.
        
        Args:
            shape: Shape of the tensor
            lam: Rate parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random Poisson values
        """
        pass
    
    @abstractmethod
    def random_exponential(self, shape: Shape, scale: float = 1.0,
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Create a tensor with random values from an exponential distribution.
        
        Args:
            shape: Shape of the tensor
            scale: Scale parameter
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random exponential values
        """
        pass
    
    @abstractmethod
    def random_categorical(self, logits: Any, num_samples: int,
                          dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Draw samples from a categorical distribution.
        
        Args:
            logits: 2D tensor with unnormalized log probabilities
            num_samples: Number of samples to draw
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with random categorical values
        """
        pass
    
    @abstractmethod
    def shuffle(self, x: Any) -> Any:
        """
        Randomly shuffle a tensor along its first dimension.
        
        Args:
            x: Input tensor
            
        Returns:
            Shuffled tensor
        """
        pass
    
    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        pass
    
    @abstractmethod
    def get_seed(self) -> Optional[int]:
        """
        Get the current random seed.
        
        Returns:
            Current random seed or None if not set
        """
        pass
    
    @abstractmethod
    def random_permutation(self, n: int, dtype: Optional[DType] = None, device: Optional[str] = None) -> Any:
        """
        Randomly permute a sequence of integers from 0 to n-1.
        
        Args:
            n: Upper bound for the range of integers to permute
            dtype: Optional data type
            device: Optional device to place the tensor on
            
        Returns:
            Tensor with a random permutation of integers from 0 to n-1
        """
        pass