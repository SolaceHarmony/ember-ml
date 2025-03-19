"""
NumPy implementation of vector operations.

This module provides NumPy implementations of vector operations.
"""

import numpy as np
from typing import Optional, Tuple, Any, Sequence

from ember_ml.backend.numpy.tensor import NumpyTensor
from ember_ml.backend.numpy.math_ops import NumpyMathOps

convert_to_tensor = NumpyTensor().convert_to_tensor
pi = NumpyMathOps().pi

class NumpyVectorOps:
    """NumPy implementation of vector operations."""

    def fft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """One dimensional discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.fft(a_tensor, n=n, axis=axis)

    def ifft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """One dimensional inverse discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.ifft(a_tensor, n=n, axis=axis)

    def fft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """Two dimensional discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.fft2(a_tensor, s=s, axes=axes)

    def ifft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """Two dimensional inverse discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.ifft2(a_tensor, s=s, axes=axes)

    def fftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """N-dimensional discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.fftn(a_tensor, s=s, axes=axes)

    def ifftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """N-dimensional inverse discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.ifftn(a_tensor, s=s, axes=axes)

    def rfft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """One dimensional discrete Fourier Transform for real input."""
        a_tensor = convert_to_tensor(a)
        return np.fft.rfft(a_tensor, n=n, axis=axis)

    def irfft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """One dimensional inverse discrete Fourier Transform for real input."""
        a_tensor = convert_to_tensor(a)
        return np.fft.irfft(a_tensor, n=n, axis=axis)

    def rfft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """Two dimensional real discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.rfft2(a_tensor, s=s, axes=axes)

    def irfft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """Two dimensional inverse real discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.irfft2(a_tensor, s=s, axes=axes)

    def rfftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """N-dimensional real discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.rfftn(a_tensor, s=s, axes=axes)

    def irfftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """N-dimensional inverse real discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return np.fft.irfftn(a_tensor, s=s, axes=axes)

    def normalize_vector(self, vector: Any) -> Any:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        vector_tensor = convert_to_tensor(vector)
        norm = np.linalg.norm(vector_tensor)
        if norm > 0:
            return vector_tensor / norm
        return vector_tensor

    def compute_energy_stability(self, wave: Any, window_size: int = 100) -> float:
        """
        Compute the energy stability of a wave signal.

        Args:
            wave: Wave signal
            window_size: Window size for stability computation

        Returns:
            Energy stability metric
        """
        wave_tensor = convert_to_tensor(wave)
        if len(wave_tensor) < window_size:
            return 1.0  # Perfectly stable for short signals

        # Compute energy in windows
        num_windows = len(wave_tensor) // window_size
        energies = []

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = wave_tensor[start:end]
            energy = np.sum(np.square(window))
            energies.append(energy)

        # Compute stability as inverse of energy variance
        if len(energies) <= 1:
            return 1.0

        energy_mean = np.mean(energies)
        if energy_mean == 0:
            return 1.0

        energy_var = np.var(energies)
        stability = 1.0 / (1.0 + energy_var / energy_mean)

        return float(stability)

    def compute_interference_strength(self, wave1: Any, wave2: Any) -> float:
        """
        Compute the interference strength between two wave signals.

        Args:
            wave1: First wave signal
            wave2: Second wave signal

        Returns:
            Interference strength metric
        """
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(len(wave1_tensor), len(wave2_tensor))
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute correlation
        correlation = np.corrcoef(wave1_tensor, wave2_tensor)[0, 1]

        # Compute phase difference
        fft1 = np.fft.fft(wave1_tensor)
        fft2 = np.fft.fft(wave2_tensor)
        phase1 = np.angle(fft1)
        phase2 = np.angle(fft2)
        phase_diff = np.abs(phase1 - phase2)
        mean_phase_diff = np.mean(phase_diff)

        # Normalize phase difference to [0, 1]
        normalized_phase_diff = mean_phase_diff / pi

        # Compute interference strength
        interference_strength = correlation * (1.0 - normalized_phase_diff)

        return float(interference_strength)

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
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(len(wave1_tensor), len(wave2_tensor))
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute FFT
        fft1 = np.fft.fft(wave1_tensor)
        fft2 = np.fft.fft(wave2_tensor)

        # Get phases
        phase1 = np.angle(fft1)
        phase2 = np.angle(fft2)

        # Get frequencies
        freqs = np.fft.fftfreq(len(wave1_tensor))

        # Apply frequency range filter if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            freq_mask = (np.abs(freqs) >= min_freq) & (np.abs(freqs) <= max_freq)
            phase1 = phase1[freq_mask]
            phase2 = phase2[freq_mask]

        # Compute phase difference
        phase_diff = phase1 - phase2

        # Compute phase coherence using circular statistics
        # Convert phase differences to complex numbers on the unit circle
        complex_phase = np.exp(1j * phase_diff)

        # Compute mean vector length (phase coherence)
        coherence = np.abs(np.mean(complex_phase))

        return float(coherence)

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
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(len(wave1_tensor), len(wave2_tensor))
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute number of windows
        num_windows = min_length - window_size + 1

        # Initialize result array using convert_to_tensor
        interference = convert_to_tensor([0.0] * num_windows)

        # Compute interference for each window
        for i in range(num_windows):
            window1 = wave1_tensor[i:i + window_size]
            window2 = wave2_tensor[i:i + window_size]

            # Compute correlation
            correlation = np.corrcoef(window1, window2)[0, 1]

            # Compute phase difference
            fft1 = np.fft.fft(window1)
            fft2 = np.fft.fft(window2)
            phase1 = np.angle(fft1)
            phase2 = np.angle(fft2)
            phase_diff = np.abs(phase1 - phase2)
            mean_phase_diff = np.mean(phase_diff)

            # Normalize phase difference to [0, 1]
            normalized_phase_diff = mean_phase_diff / pi

            # Compute interference strength
            interference[i] = correlation * (1.0 - normalized_phase_diff)

        return interference

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
        return np.sqrt(np.sum(np.square(x_tensor - y_tensor)))

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
        dot_product = np.sum(x_tensor * y_tensor)
        norm_x = np.sqrt(np.sum(np.square(x_tensor)))
        norm_y = np.sqrt(np.sum(np.square(y_tensor)))
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
        initial_tensor = convert_to_tensor(initial_value)
        decay_tensor = convert_to_tensor(decay_rate)
        time_tensor = convert_to_tensor(time_step)
        return initial_tensor * np.exp(-decay_tensor * time_tensor)

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
        return np.exp(-0.5 * np.square((x_tensor - mu_tensor) / sigma_tensor)) / (sigma_tensor * np.sqrt(2.0 * pi))