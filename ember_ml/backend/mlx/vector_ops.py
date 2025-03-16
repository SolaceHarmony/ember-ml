"""
MLX implementation of vector operations.

This module provides MLX implementations of vector operations.
"""

import mlx.core as mx
from typing import Optional, Tuple, Any, Sequence

from ember_ml.backend.mlx.tensor import MLXTensor
from ember_ml.backend.mlx.math_ops import MLXMathOps

convert_to_tensor = MLXTensor().convert_to_tensor
pi = MLXMathOps().pi

class MLXVectorOps:
    """MLX implementation of vector operations."""

    def normalize_vector(self, vector: Any) -> Any:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        vector_tensor = convert_to_tensor(vector)
        norm = mx.linalg.norm(vector_tensor)

        # Avoid division by zero
        if norm > 0:
            return vector_tensor / norm
        return vector_tensor

    def compute_energy_stability(self, wave: Any, window_size: int = 100) -> float:
        """
        Compute the energy stability of a wave signal.
        """
        wave_tensor = convert_to_tensor(wave)

        if len(wave_tensor.shape) == 0 or wave_tensor.shape[0] < window_size:
            return 1.0  # Perfectly stable for short signals

        # Compute energy in windows
        num_windows = wave_tensor.shape[0] // window_size
        energies = []

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = wave_tensor[start:end]
            energy = mx.sum(mx.square(window))
            energies.append(energy.item())

        # Compute stability as inverse of energy variance
        if len(energies) <= 1:
            return 1.0

        energies_tensor = convert_to_tensor(energies)
        energy_mean = mx.mean(energies_tensor)

        if energy_mean == 0:
            return 1.0

        energy_var = mx.var(energies_tensor)
        stability = 1.0 / (1.0 + energy_var / energy_mean)

        # Safe conversion to float
        stability_val = mx.array(stability).item()
        return float(str(stability_val))

    def compute_interference_strength(self, wave1: Any, wave2: Any) -> float:
        """
        Compute the interference strength between two wave signals.
        """
        # Convert to MLX tensors
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute correlation
        wave1_mean = mx.mean(wave1_tensor)
        wave2_mean = mx.mean(wave2_tensor)
        wave1_centered = wave1_tensor - wave1_mean
        wave2_centered = wave2_tensor - wave2_mean

        numerator = mx.sum(wave1_centered * wave2_centered)
        denominator = mx.sqrt(mx.sum(wave1_centered * wave1_centered)) * mx.sqrt(mx.sum(wave2_centered * wave2_centered))
        denominator = denominator + convert_to_tensor(1e-8)  # Add epsilon as MLX tensor
        correlation = numerator / denominator

        # Compute phase difference using FFT
        fft1 = mx.fft.fft(wave1_tensor)
        fft2 = mx.fft.fft(wave2_tensor)

        # MLX doesn't have angle() directly, compute phase using arctan2
        phase1 = mx.arctan2(mx.imag(fft1), mx.real(fft1))
        phase2 = mx.arctan2(mx.imag(fft2), mx.real(fft2))
        phase_diff = mx.abs(phase1 - phase2)
        mean_phase_diff = mx.mean(phase_diff)

        # Normalize phase difference to [0, 1]
        pi_tensor = convert_to_tensor(MLXMathOps.pi)
        normalized_phase_diff = mean_phase_diff / pi_tensor

        # Compute interference strength
        interference_strength = correlation * (1.0 - normalized_phase_diff)

        # Safe conversion to float
        val = mx.array(interference_strength).item()
        return float(str(val))

    def compute_phase_coherence(self, wave1: Any, wave2: Any, freq_range: Optional[Tuple[float, float]] = None) -> float:
        """
        Compute the phase coherence between two wave signals.
        """
        # Convert to MLX tensors
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute FFT using MLX
        fft1 = mx.fft.fft(wave1_tensor)
        fft2 = mx.fft.fft(wave2_tensor)

        # Get phases using arctan2 since angle() isn't available
        phase1 = mx.arctan2(mx.imag(fft1), mx.real(fft1))
        phase2 = mx.arctan2(mx.imag(fft2), mx.real(fft2))

        # Compute frequencies using manual calculation since fftfreq isn't available
        n = wave1_tensor.shape[0]
        freqs = mx.arange(n) / n

        # Apply frequency range filter if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
            phase1 = mx.where(freq_mask, phase1, mx.zeros_like(phase1))
            phase2 = mx.where(freq_mask, phase2, mx.zeros_like(phase2))

        # Compute phase difference
        phase_diff = phase1 - phase2

        # Use Euler's formula for complex phase calculation
        complex_real = mx.cos(phase_diff)
        complex_imag = mx.sin(phase_diff)
        coherence = mx.sqrt(mx.mean(complex_real) ** 2 + mx.mean(complex_imag) ** 2)

        # Safe conversion to float
        val = mx.array(coherence).item()
        return float(str(val))

    def partial_interference(self, wave1: Any, wave2: Any, window_size: int = 100) -> Any:
        """
        Compute the partial interference between two wave signals over sliding windows.
        """
        # Convert to MLX tensors
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute number of windows
        num_windows = min_length - window_size + 1

        # Initialize result array
        interference = mx.zeros(num_windows)
        result = []

        # Compute interference for each window
        for i in range(num_windows):
            window1 = wave1_tensor[i:i+window_size]
            window2 = wave2_tensor[i:i+window_size]

            # Compute correlation
            window1_centered = window1 - mx.mean(window1)
            window2_centered = window2 - mx.mean(window2)
            correlation = mx.sum(window1_centered * window2_centered) / (
                mx.sqrt(mx.sum(window1_centered ** 2)) * mx.sqrt(mx.sum(window2_centered ** 2)) + 1e-8
            )

            # Compute FFT for this window
            fft1 = mx.fft.fft(window1)
            fft2 = mx.fft.fft(window2)

            # Get phases using arctan2
            phase1 = mx.arctan2(mx.imag(fft1), mx.real(fft1))
            phase2 = mx.arctan2(mx.imag(fft2), mx.real(fft2))
            phase_diff = mx.abs(phase1 - phase2)
            mean_phase_diff = mx.mean(phase_diff)

            # Normalize phase difference to [0, 1]
            normalized_phase_diff = mean_phase_diff / MLXMathOps.pi

            # Compute interference strength
            val = correlation * (1.0 - normalized_phase_diff)
            result.append(mx.array(val).item())

        return mx.array(result)

    def euclidean_distance(self, x: Any, y: Any) -> Any:
        """
        Compute the Euclidean distance between two vectors.

        Args:
            x: First vector
            y: Second vector

        Returns:
            Euclidean distance
        """
        x_tensor = convert_to_tensor(x)
        y_tensor = convert_to_tensor(y)

        return mx.sqrt(mx.sum(mx.square(x_tensor - y_tensor)))

    def cosine_similarity(self, x: Any, y: Any) -> Any:
        """
        Compute the cosine similarity between two vectors.

        Args:
            x: First vector
            y: Second vector

        Returns:
            Cosine similarity
        """
        x_tensor = convert_to_tensor(x)
        y_tensor = convert_to_tensor(y)

        dot_product = mx.sum(x_tensor * y_tensor)
        norm_x = mx.sqrt(mx.sum(mx.square(x_tensor)))
        norm_y = mx.sqrt(mx.sum(mx.square(y_tensor)))
        return dot_product / (norm_x * norm_y + 1e-8)

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
        initial_value_tensor = convert_to_tensor(initial_value)
        decay_rate_tensor = convert_to_tensor(decay_rate)
        time_step_tensor = convert_to_tensor(time_step)

        return initial_value_tensor * mx.exp(-decay_rate_tensor * time_step_tensor)

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
        x_tensor = convert_to_tensor(x)
        mu_tensor = convert_to_tensor(mu)
        sigma_tensor = convert_to_tensor(sigma)

        return mx.exp(-0.5 * mx.square((x_tensor - mu_tensor) / sigma_tensor)) / (sigma_tensor * mx.sqrt(mx.array(2.0)) * mx.sqrt(MLXMathOps.pi))

    def fft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """
        One dimensional discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)
        return mx.fft.fft(a_tensor, n=n, axis=axis)

    def ifft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """
        One dimensional inverse discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)
        return mx.fft.ifft(a_tensor, n=n, axis=axis)

    def fft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """
        Two dimensional discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)

        # MLX doesn't have direct fft2, so we implement it using sequential 1D FFTs
        result = a_tensor
        for axis in axes:
            shape_i = s[axes.index(axis)] if s is not None else None
            result = mx.fft.fft(result, n=shape_i, axis=axis)
        return result

    def ifft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """
        Two dimensional inverse discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)

        # MLX doesn't have direct ifft2, so we implement it using sequential 1D IFFTs
        result = a_tensor
        for axis in axes:
            shape_i = s[axes.index(axis)] if s is not None else None
            result = mx.fft.ifft(result, n=shape_i, axis=axis)
        return result

    def fftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        N-dimensional discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)

        # If axes not specified, transform over all axes
        if axes is None:
            axes = tuple(range(a_tensor.ndim))

        # MLX doesn't have direct fftn, so we implement it using sequential 1D FFTs
        result = a_tensor
        for i, axis in enumerate(axes):
            shape_i = s[i] if s is not None else None
            result = mx.fft.fft(result, n=shape_i, axis=axis)
        return result

    def ifftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        N-dimensional inverse discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)

        # If axes not specified, transform over all axes
        if axes is None:
            axes = tuple(range(a_tensor.ndim))

        # MLX doesn't have direct ifftn, so we implement it using sequential 1D IFFTs
        result = a_tensor
        for i, axis in enumerate(axes):
            shape_i = s[axes.index(axis)] if s is not None else None
            result = mx.fft.ifft(result, n=shape_i, axis=axis)
        return result

    def rfft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """
        One dimensional discrete Fourier Transform for real input.
        """
        a_tensor = convert_to_tensor(a)
        # MLX doesn't have direct rfft, so we implement it using regular fft
        # and taking only the non-redundant half
        result = mx.fft.fft(a_tensor, n=n, axis=axis)
        input_size = a_tensor.shape[axis]
        output_size = (input_size if n is None else n) // 2 + 1

        slices = [slice(None)] * result.ndim
        slices[axis] = slice(0, output_size)
        return result[tuple(slices)]

    def irfft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """
        One dimensional inverse discrete Fourier Transform for real input.
        """
        a_tensor = convert_to_tensor(a)

        # Handle the size parameter
        input_shape = a_tensor.shape
        if n is None:
            n = 2 * (input_shape[axis] - 1)

        # Reconstruct conjugate symmetric Fourier components
        # Take advantage of conjugate symmetry: F(-f) = F(f)*
        middle_slices = [slice(None)] * a_tensor.ndim
        middle_slices[axis] = slice(1, -1)

        if input_shape[axis] > 1:
            # Since MLX doesn't have flip, we'll reverse the array using array indexing
            rev_idx = mx.arange(input_shape[axis] - 2, -1, -1)

            # Create broadcasted indices for other dimensions
            rev_shape = [1] * a_tensor.ndim
            rev_shape[axis] = rev_idx.shape[0]
            rev_idx = rev_idx.reshape(rev_shape)

            # Broadcast rev_idx to match tensor shape for other dimensions
            broadcast_shape = list(input_shape)
            broadcast_shape[axis] = rev_idx.shape[axis]
            rev_idx = mx.broadcast_to(rev_idx, broadcast_shape)

            # Get reversed data
            reversed_data = mx.take(a_tensor[tuple(middle_slices)], rev_idx, axis=axis)

            # Concatenate with conjugate
            full_fourier = mx.concatenate([
                a_tensor,
                mx.conj(reversed_data)
            ], axis=axis)
        else:
            full_fourier = a_tensor

        # Perform inverse FFT
        return mx.real(mx.fft.ifft(full_fourier, n=n, axis=axis))

    def rfft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """
        Two dimensional real discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)

        # MLX doesn't have rfft2, so implement using sequential 1D rffts
        result = a_tensor
        for i, axis in enumerate(axes):
            shape_i = s[i] if s is not None else None
            if i == 0:
                # First axis - use rfft to get real FFT
                result = self.rfft(result, n=shape_i, axis=axis)
            else:
                # Second axis - use regular fft since data is already complex
                result = mx.fft.fft(result, n=shape_i, axis=axis)
        return result

    def irfft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """
        Two dimensional inverse real discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)

        # MLX doesn't have irfft2, so implement using sequential 1D iffts
        result = a_tensor
        for i, axis in enumerate(reversed(axes)):
            shape_i = s[-i-1] if s is not None else None
            if i == 0:
                # Last axis - use regular ifft since data is complex
                result = mx.fft.ifft(result, n=shape_i, axis=axis)
            else:
                # First axis - use irfft to get real result
                result = self.irfft(result, n=shape_i, axis=axis)
        return result

    def rfftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        N-dimensional real discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)

        # If axes not specified, transform over all axes
        if axes is None:
            axes = tuple(range(a_tensor.ndim))

        # MLX doesn't have rfftn, so implement using sequential 1D ffts
        result = a_tensor
        for i, axis in enumerate(axes):
            shape_i = s[i] if s is not None else None
            if i == 0:
                # First axis - use rfft to get real FFT
                result = self.rfft(result, n=shape_i, axis=axis)
            else:
                # Other axes - use regular fft since data is already complex
                result = mx.fft.fft(result, n=shape_i, axis=axis)
        return result

    def irfftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """
        N-dimensional inverse real discrete Fourier Transform.
        """
        a_tensor = convert_to_tensor(a)

        # If axes not specified, transform over all axes
        if axes is None:
            axes = tuple(range(a_tensor.ndim))

        # MLX doesn't have irfftn, so implement using sequential 1D iffts
        result = a_tensor
        for i, axis in enumerate(reversed(axes)):
            shape_i = s[-i-1] if s is not None else None
            if i == 0:
                # Last axis - use regular ifft since data is complex
                result = mx.fft.ifft(result, n=shape_i, axis=axis)
            else:
                # Other axes - use irfft to get real result
                result = self.irfft(result, n=shape_i, axis=axis)
        return result