"""
Vector operations interface.

This module defines the abstract interface for vector operations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional, Any, Tuple

# Type aliases
Shape = Union[int, Sequence[int]]

class VectorOps(ABC):
    """Abstract interface for vector operations."""
    
    @abstractmethod
    def normalize_vector(self, vector: Any) -> Any:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        pass
    
    @abstractmethod
    def compute_energy_stability(self, wave: Any, window_size: int = 100) -> float:
        """
        Compute the energy stability of a wave signal.
        
        Args:
            wave: Wave signal
            window_size: Window size for stability computation
            
        Returns:
            Energy stability metric
        """
        pass
    
    @abstractmethod
    def compute_interference_strength(self, wave1: Any, wave2: Any) -> float:
        """
        Compute the interference strength between two wave signals.
        
        Args:
            wave1: First wave signal
            wave2: Second wave signal
            
        Returns:
            Interference strength metric
        """
        pass
    
    @abstractmethod
    def compute_phase_coherence(self, wave1: Any, wave2: Any, freq_range: Optional[Tuple[float, float]] = None) -> float:
        """
        Compute the phase coherence between two wave signals.
        
        Args:
            wave1: First wave signal
            wave2: Second wave signal
            freq_range: Optional frequency range to consider (min_freq, max_freq)
            
        Returns:
            Phase coherence metric
        """
        pass
    
    @abstractmethod
    def partial_interference(self, wave1: Any, wave2: Any, window_size: int = 100) -> Any:
        """
        Compute the partial interference between two wave signals over sliding windows.
        
        Args:
            wave1: First wave signal
            wave2: Second wave signal
            window_size: Size of the sliding window
            
        Returns:
            Array of partial interference values for each window
        """
        pass
    
    @abstractmethod
    def euclidean_distance(self, x: Any, y: Any) -> Any:
        """
        Compute the Euclidean distance between two vectors.
        
        Args:
            x: First vector
            y: Second vector
            
        Returns:
            Euclidean distance
        """
        pass
    
    @abstractmethod
    def cosine_similarity(self, x: Any, y: Any) -> Any:
        """
        Compute the cosine similarity between two vectors.
        
        Args:
            x: First vector
            y: Second vector
            
        Returns:
            Cosine similarity
        """
        pass
    
    @abstractmethod
    def exponential_decay(self, initial_value: Any, decay_rate: Any, time_step: Any) -> Any:
        """
        Compute exponential decay.
        
        Args:
            initial_value: Initial value
            decay_rate: Decay rate
            time_step: Time step
            
        Returns:
            Exponentially decayed value
        """
        pass
    
    @abstractmethod
    def gaussian(self, x: Any, mu: Any = 0.0, sigma: Any = 1.0) -> Any:
        """
        Compute the Gaussian function.
        
        Args:
            x: Input tensor
            mu: Mean
            sigma: Standard deviation
            
        Returns:
            Gaussian function evaluated at x
        """
        pass
        
    @abstractmethod
    def fft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """
        One dimensional discrete Fourier Transform.
        
        Args:
            a: Input array
            n: Length of transformed axis. If n < a.shape[axis], a is truncated
               If n > a.shape[axis], a is zero-padded
            axis: Axis over which to compute FFT
            
        Returns:
            The transformed input array
        """
        pass
        
    @abstractmethod
    def ifft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """
        One dimensional inverse discrete Fourier Transform.
        
        Args:
            a: Input array
            n: Length of transformed axis. If n < a.shape[axis], a is truncated
               If n > a.shape[axis], a is zero-padded
            axis: Axis over which to compute inverse FFT
            
        Returns:
            The inverse transformed input array
        """
        pass
        
    @abstractmethod
    def fft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """
        Two dimensional discrete Fourier Transform.
        
        Args:
            a: Input array
            s: Shape of the result (n, m). If given, each dimension of a is zero or 
               truncated to match s
            axes: Axes over which to compute 2D FFT
            
        Returns:
            The transformed input array
        """
        pass
        
    @abstractmethod
    def ifft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """
        Two dimensional inverse discrete Fourier Transform.
        
        Args:
            a: Input array
            s: Shape of the result (n, m). If given, each dimension of a is zero or
               truncated to match s
            axes: Axes over which to compute inverse 2D FFT
            
        Returns:
            The inverse transformed input array
        """
        pass
        
    @abstractmethod
    def fftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        N-dimensional discrete Fourier Transform.
        
        Args:
            a: Input array
            s: Shape of the result. If given, each dimension of a is zero or
               truncated to match s
            axes: Axes over which to compute the N-D FFT
            
        Returns:
            The transformed input array
        """
        pass
        
    @abstractmethod
    def ifftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        N-dimensional inverse discrete Fourier Transform.
        
        Args:
            a: Input array
            s: Shape of the result. If given, each dimension of a is zero or
               truncated to match s
            axes: Axes over which to compute the inverse N-D FFT
            
        Returns:
            The inverse transformed input array
        """
        pass
        
    @abstractmethod
    def rfft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """
        One dimensional discrete Fourier Transform for real input.
        
        Args:
            a: Input array (real)
            n: Length of transformed axis. If n < a.shape[axis], a is truncated
               If n > a.shape[axis], a is zero-padded
            axis: Axis over which to compute real FFT
            
        Returns:
            The transformed input array
        """
        pass
        
    @abstractmethod
    def irfft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """
        One dimensional inverse discrete Fourier Transform for real input.
        
        Args:
            a: Input array
            n: Length of transformed axis. If n < a.shape[axis], a is truncated
               If n > a.shape[axis], a is zero-padded
            axis: Axis over which to compute inverse real FFT
            
        Returns:
            The inverse transformed input array (real)
        """
        pass
    
    @abstractmethod
    def rfft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """
        Two dimensional real discrete Fourier Transform.
        
        Args:
            a: Input array (real)
            s: Shape of the result (n, m). If given, each dimension of a is zero or 
               truncated to match s
            axes: Axes over which to compute 2D real FFT
            
        Returns:
            The transformed input array
        """
        pass
        
    @abstractmethod
    def irfft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """
        Two dimensional inverse real discrete Fourier Transform.
        
        Args:
            a: Input array
            s: Shape of the result (n, m). If given, each dimension of a is zero or
               truncated to match s
            axes: Axes over which to compute inverse 2D real FFT
            
        Returns:
            The inverse transformed input array (real)
        """
        pass
        
    @abstractmethod
    def rfftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        N-dimensional real discrete Fourier Transform.
        
        Args:
            a: Input array (real)
            s: Shape of the result. If given, each dimension of a is zero or
               truncated to match s
            axes: Axes over which to compute the N-D real FFT
            
        Returns:
            The transformed input array
        """
        pass
        
    @abstractmethod
    def irfftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        N-dimensional inverse real discrete Fourier Transform.
        
        Args:
            a: Input array
            s: Shape of the result. If given, each dimension of a is zero or
               truncated to match s
            axes: Axes over which to compute the inverse N-D real FFT
            
        Returns:
            The inverse transformed input array (real)
        """
        pass