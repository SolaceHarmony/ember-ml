"""
Vector operations interface.

This module defines the abstract interface for vector operations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional, Any, Tuple
# Import TensorLike for more specific type hinting (assuming a base definition exists)
# If not, we might need to define a Protocol or use Any/object


# Type aliases
Shape = Union[int, Sequence[int]]

class VectorOps(ABC):
    """Abstract interface for vector operations."""
    
    @abstractmethod
    def normalize_vector(self, input_vector: Any) -> Any:
        """
        Normalize an input vector to unit length (L2 norm).

        If the vector's norm is zero, the original vector should be returned.

        Args:
            input_vector: The vector to normalize.

        Returns:
            The normalized vector, specific to the backend implementation.
        """
        pass
    
    @abstractmethod
    def compute_energy_stability(self, input_wave: Any, window_size: int = 100) -> float:
        """
        Compute the energy stability of a wave signal.

        Calculates stability based on the variance of energy across sliding windows.
        A value closer to 1.0 indicates higher stability.

        Args:
            input_wave: The input wave signal.
            window_size: The size of the sliding window used for energy calculation.

        Returns:
            A float representing the energy stability metric (0.0 to 1.0).
        """
        pass
    
    @abstractmethod
    def compute_interference_strength(self, input_wave1: Any, input_wave2: Any) -> float:
        """
        Compute the interference strength between two wave signals.

        Combines correlation and phase difference to quantify interference.
        A value closer to 1 suggests strong constructive interference,
        closer to -1 suggests strong destructive interference, and near 0
        suggests low interference or incoherence.

        Args:
            input_wave1: The first input wave signal.
            input_wave2: The second input wave signal.

        Returns:
            A float representing the interference strength metric.
        """
        pass
    
    @abstractmethod
    def compute_phase_coherence(self, input_wave1: Any, input_wave2: Any, freq_range: Optional[Tuple[float, float]] = None) -> float:
        """
        Compute the phase coherence between two wave signals.

        Calculates the consistency of the phase difference between two signals,
        optionally within a specified frequency range. Uses circular statistics.
        A value closer to 1 indicates high phase coherence.

        Args:
            input_wave1: The first input wave signal.
            input_wave2: The second input wave signal.
            freq_range: Optional tuple (min_freq, max_freq) to filter frequencies
                        before computing coherence.

        Returns:
            A float representing the phase coherence metric (0.0 to 1.0).
        """
        pass
    
    @abstractmethod
    def partial_interference(self, input_wave1: Any, input_wave2: Any, window_size: int = 100) -> Any:
        """
        Compute the partial interference between two wave signals over sliding windows.

        Calculates interference strength for overlapping windows of the signals.

        Args:
            input_wave1: The first input wave signal.
            input_wave2: The second input wave signal.
            window_size: The size of the sliding window.

        Returns:
            An array containing the interference strength for each window.
        """
        pass
    
    @abstractmethod
    def euclidean_distance(self, vector1: Any, vector2: Any) -> Any:
        """
        Compute the Euclidean (L2) distance between two vectors.

        Args:
            vector1: The first input vector.
            vector2: The second input vector.

        Returns:
            A scalar representing the Euclidean distance.
        """
        pass
    
    @abstractmethod
    def cosine_similarity(self, vector1: Any, vector2: Any) -> Any:
        """
        Compute the cosine similarity between two vectors.

        Measures the cosine of the angle between two non-zero vectors.
        Result ranges from -1 (exactly opposite) to 1 (exactly the same).

        Args:
            vector1: The first input vector.
            vector2: The second input vector.

        Returns:
            A scalar representing the cosine similarity.
        """
        pass
    
    @abstractmethod
    def exponential_decay(self, initial_value: Any, decay_rate: Any, time_step: Optional[Any] = None) -> Any:
        """
        Compute exponential decay.

        If `time_step` is provided, applies uniform decay:
            value = initial * exp(-rate * time_step)
        If `time_step` is None, applies index-based decay to the input array:
            value[i] = initial[i] * exp(-rate * i)

        Args:
            initial_value: The starting value(s).
            decay_rate: The rate of decay (must be positive).
            time_step: The elapsed time for uniform decay, or None for index-based.
                       Defaults to None (index-based).

        Returns:
            The value(s) after exponential decay.
        """
        pass
    
    @abstractmethod
    def fft(self, input_array: Any, output_length: Optional[int] = None, axis: int = -1) -> Any:
        """
        Compute the one-dimensional discrete Fourier Transform.

        Args:
            input_array: Input array.
            output_length: Length of the transformed axis of the output.
               If None, the length of the input along the axis is used.
            axis: Axis over which to compute the FFT.

        Returns:
            The transformed array.
        """
        pass
        
    @abstractmethod
    def ifft(self, input_array: Any, output_length: Optional[int] = None, axis: int = -1) -> Any:
        """
        Compute the one-dimensional inverse discrete Fourier Transform.

        Args:
            input_array: Input array.
            output_length: Length of the transformed axis of the output.
               If None, the length of the input along the axis is used.
            axis: Axis over which to compute the inverse FFT.

        Returns:
            The inverse transformed array.
        """
        pass
        
    @abstractmethod
    def fft2(self, input_array: Any, output_shape: Optional[Shape] = None, axes: Optional[Tuple[int, int]] = (-2, -1)) -> Any:
        """
        Compute the two-dimensional discrete Fourier Transform.

        Args:
            input_array: Input array.
            output_shape: Shape (length of each transformed axis) of the output.
               If None, the shape of the input along the axes is used.
            axes: Axes over which to compute the FFT. Defaults to the last two axes.

        Returns:
            The transformed array.
        """
        pass
        
    @abstractmethod
    def ifft2(self, input_array: Any, output_shape: Optional[Shape] = None, axes: Optional[Tuple[int, int]] = (-2, -1)) -> Any:
        """
        Compute the two-dimensional inverse discrete Fourier Transform.

        Args:
            input_array: Input array.
            output_shape: Shape (length of each transformed axis) of the output.
               If None, the shape of the input along the axes is used.
            axes: Axes over which to compute the inverse FFT. Defaults to the last two axes.

        Returns:
            The inverse transformed array.
        """
        pass
        
    @abstractmethod
    def fftn(self, input_array: Any, output_shape: Optional[Shape] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        Compute the N-dimensional discrete Fourier Transform.

        Args:
            input_array: Input array.
            output_shape: Shape (length of each transformed axis) of the output.
               If None, the shape of the input along the axes is used.
            axes: Axes over which to compute the FFT. If None, all axes are used.

        Returns:
            The transformed array.
        """
        pass
        
    @abstractmethod
    def ifftn(self, input_array: Any, output_shape: Optional[Shape] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        Compute the N-dimensional inverse discrete Fourier Transform.

        Args:
            input_array: Input array.
            output_shape: Shape (length of each transformed axis) of the output.
               If None, the shape of the input along the axes is used.
            axes: Axes over which to compute the inverse FFT. If None, all axes are used.

        Returns:
            The inverse transformed array.
        """
        pass
        
    @abstractmethod
    def rfft(self, input_array: Any, output_length: Optional[int] = None, axis: int = -1) -> Any:
        """
        Compute the one-dimensional discrete Fourier Transform for real input.

        Args:
            input_array: Input array (must be real).
            output_length: Length of the transformed axis of the output.
               If None, the length of the input along the axis is used.
            axis: Axis over which to compute the FFT.

        Returns:
            The transformed array.
        """
        pass
        
    @abstractmethod
    def irfft(self, input_array: Any, output_length: Optional[int] = None, axis: int = -1) -> Any:
        """
        Compute the one-dimensional inverse discrete Fourier Transform for real input.

        Args:
            input_array: Input array.
            output_length: Length of the output array along the transformed axis.
               If None, uses default logic based on input shape.
            axis: Axis over which to compute the inverse FFT.

        Returns:
            The inverse transformed array (real).
        """
        pass
    
    @abstractmethod
    def rfft2(self, input_array: Any, output_shape: Optional[Shape] = None, axes: Optional[Tuple[int, int]] = (-2, -1)) -> Any:
        """
        Compute the two-dimensional discrete Fourier Transform for real input.

        Args:
            input_array: Input array (must be real).
            output_shape: Shape (length of each transformed axis) of the output.
               If None, the shape of the input along the axes is used.
            axes: Axes over which to compute the FFT. Defaults to the last two axes.

        Returns:
            The transformed array.
        """
        pass
        
    @abstractmethod
    def irfft2(self, input_array: Any, output_shape: Optional[Shape] = None, axes: Optional[Tuple[int, int]] = (-2, -1)) -> Any:
        """
        Compute the two-dimensional inverse discrete Fourier Transform for real input.

        Args:
            input_array: Input array.
            output_shape: Shape (length of each transformed axis) of the output.
               If None, the shape of the input along the axes is used.
            axes: Axes over which to compute the inverse FFT. Defaults to the last two axes.

        Returns:
            The inverse transformed array (real).
        """
        pass
        
    @abstractmethod
    def rfftn(self, input_array: Any, output_shape: Optional[Shape] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        Compute the N-dimensional discrete Fourier Transform for real input.

        Args:
            input_array: Input array (must be real).
            output_shape: Shape (length of each transformed axis) of the output.
               If None, the shape of the input along the axes is used.
            axes: Axes over which to compute the FFT. If None, all axes are used.

        Returns:
            The transformed array.
        """
        pass
        
    @abstractmethod
    def irfftn(self, input_array: Any, output_shape: Optional[Shape] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        Compute the N-dimensional inverse discrete Fourier Transform for real input.

        Args:
            input_array: Input array.
            output_shape: Shape (length of each transformed axis) of the output.
               If None, the shape of the input along the axes is used.
            axes: Axes over which to compute the inverse FFT. If None, all axes are used.

        Returns:
            The inverse transformed array (real).
        """
        pass