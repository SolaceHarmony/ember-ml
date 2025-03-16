"""
PyTorch implementation of vector operations.

This module provides PyTorch implementations of vector operations.
"""

import torch
from typing import Optional, Tuple, Any, Sequence

from ember_ml.backend.torch.tensor import TorchTensor
from ember_ml.backend.torch.math_ops import TorchMathOps

convert_to_tensor = TorchTensor().convert_to_tensor
pi = TorchMathOps().pi

class TorchVectorOps:
    """PyTorch implementation of vector operations."""

    def fft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """One dimensional discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.fft(a_tensor, n=n, dim=axis)

    def ifft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """One dimensional inverse discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.ifft(a_tensor, n=n, dim=axis)

    def fft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """Two dimensional discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.fft2(a_tensor, s=s, dim=axes)

    def ifft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """Two dimensional inverse discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.ifft2(a_tensor, s=s, dim=axes)

    def fftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """N-dimensional discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.fftn(a_tensor, s=s, dim=axes)

    def ifftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """N-dimensional inverse discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.ifftn(a_tensor, s=s, dim=axes)

    def rfft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """One dimensional discrete Fourier Transform for real input."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.rfft(a_tensor, n=n, dim=axis)

    def irfft(self, a: Any, n: Optional[int] = None, axis: int = -1) -> Any:
        """One dimensional inverse discrete Fourier Transform for real input."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.irfft(a_tensor, n=n, dim=axis)

    def rfft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """Two dimensional real discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.rfft2(a_tensor, s=s, dim=axes)

    def irfft2(self, a: Any, s: Optional[Tuple[int, int]] = None, axes: Tuple[int, int] = (-2, -1)) -> Any:
        """Two dimensional inverse real discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.irfft2(a_tensor, s=s, dim=axes)

    def rfftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """N-dimensional real discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.rfftn(a_tensor, s=s, dim=axes)

    def irfftn(self, a: Any, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None) -> Any:
        """N-dimensional inverse real discrete Fourier Transform."""
        a_tensor = convert_to_tensor(a)
        return torch.fft.irfftn(a_tensor, s=s, dim=axes)

    def normalize_vector(self, vector: Any) -> Any:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        vector_tensor = convert_to_tensor(vector)
        norm = torch.norm(vector_tensor, p=2)
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
            energy = torch.sum(torch.square(window))
            energies.append(energy.item())

        # Compute stability as inverse of energy variance
        if len(energies) <= 1:
            return 1.0

        energies_tensor = convert_to_tensor(energies)
        energy_mean = torch.mean(energies_tensor)
        if energy_mean == 0:
            return 1.0

        energy_var = torch.var(energies_tensor)
        stability = 1.0 / (1.0 + energy_var / energy_mean)

        return float(stability.item())

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
        # PyTorch doesn't have a direct corrcoef function, so we compute it manually
        wave1_centered = wave1_tensor - torch.mean(wave1_tensor)
        wave2_centered = wave2_tensor - torch.mean(wave2_tensor)
        correlation = torch.sum(wave1_centered * wave2_centered) / (
            torch.sqrt(torch.sum(wave1_centered ** 2)) * torch.sqrt(torch.sum(wave2_centered ** 2)) + 1e-8
        )

        # Compute phase difference using PyTorch's FFT
        fft1 = torch.fft.fft(wave1_tensor.float())
        fft2 = torch.fft.fft(wave2_tensor.float())
        phase1 = torch.angle(fft1)
        phase2 = torch.angle(fft2)
        phase_diff = torch.abs(phase1 - phase2)
        mean_phase_diff = torch.mean(phase_diff)

        # Normalize phase difference to [0, 1]
        normalized_phase_diff = mean_phase_diff / pi

        # Compute interference strength
        # High correlation and low phase difference = constructive interference
        # Low correlation and high phase difference = destructive interference
        interference_strength = correlation.item() * (1.0 - normalized_phase_diff)

        return float(interference_strength)

    def compute_phase_coherence(self, wave1: Any, wave2: Any, freq_range: Optional[Tuple[float, float]] = None) -> float:
        """
        Compute the phase coherence between two wave signals.
        """
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(len(wave1_tensor), len(wave2_tensor))
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute FFT using PyTorch
        fft1 = torch.fft.fft(wave1_tensor.float())
        fft2 = torch.fft.fft(wave2_tensor.float())

        # Get phases
        phase1 = torch.angle(fft1)
        phase2 = torch.angle(fft2)

        # Get frequencies
        freqs = torch.fft.fftfreq(len(wave1_tensor))

        # Apply frequency range filter if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            freq_mask = (torch.abs(freqs) >= min_freq) & (torch.abs(freqs) <= max_freq)
            phase1 = phase1[freq_mask]
            phase2 = phase2[freq_mask]

        # Compute phase difference
        phase_diff = phase1 - phase2

        # Convert phase differences to complex numbers on the unit circle
        # For complex numbers in PyTorch, we need to create a complex tensor
        complex_phase = torch.exp(torch.complex(torch.zeros_like(phase_diff), phase_diff))

        # Compute mean vector length (phase coherence)
        coherence = torch.abs(torch.mean(complex_phase))

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
            window1_centered = window1 - torch.mean(window1)
            window2_centered = window2 - torch.mean(window2)
            correlation = torch.sum(window1_centered * window2_centered) / (
                torch.sqrt(torch.sum(window1_centered ** 2)) * torch.sqrt(torch.sum(window2_centered ** 2)) + 1e-8
            )

            # Compute phase difference using PyTorch
            fft1 = torch.fft.fft(window1.float())
            fft2 = torch.fft.fft(window2.float())
            phase1 = torch.angle(fft1)
            phase2 = torch.angle(fft2)
            phase_diff = torch.abs(phase1 - phase2)
            mean_phase_diff = torch.mean(phase_diff)

            # Normalize phase difference to [0, 1]
            normalized_phase_diff = mean_phase_diff / pi

            # Compute interference strength
            interference[i] = correlation.item() * (1.0 - normalized_phase_diff)

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
        return torch.sqrt(torch.sum(torch.square(x_tensor - y_tensor)))

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
        dot_product = torch.sum(x_tensor * y_tensor)
        norm_x = torch.sqrt(torch.sum(torch.square(x_tensor)))
        norm_y = torch.sqrt(torch.sum(torch.square(y_tensor)))
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
        return initial_tensor * torch.exp(-decay_tensor * time_tensor)

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
        return torch.exp(-0.5 * torch.square((x_tensor - mu_tensor) / sigma_tensor)) / (sigma_tensor * torch.sqrt(convert_to_tensor(2.0)) * torch.sqrt(pi))