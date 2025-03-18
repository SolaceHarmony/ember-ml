"""
PyTorch implementation of vector operations.

This module provides PyTorch implementations of vector operations including FFT transforms
and wave signal analysis functions.
"""

import torch
from typing import Optional, Tuple, Sequence, List, Union,TYPE_CHECKING
from torch import Tensor

from ember_ml.backend.torch.tensor import TorchTensor
from ember_ml.backend.torch.math_ops import TorchMathOps
if TYPE_CHECKING:
    from ember_ml.backend.torch.config import TensorLike, AllTensorLike

convert_to_tensor = TorchTensor().convert_to_tensor
pi = TorchMathOps().pi

class TorchVectorOps:
    """PyTorch implementation of vector operations."""

    def normalize_vector(self, vector: TensorLike) -> Tensor:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        vector_tensor = convert_to_tensor(vector)
        norm = torch.linalg.norm(vector_tensor)
        if norm > 0:
            return torch.divide(vector_tensor, norm)
        return vector_tensor

    def euclidean_distance(self, x: TensorLike, y: TensorLike) -> Tensor:
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
        return torch.sqrt(torch.sum(torch.square(torch.subtract(x_tensor, y_tensor))))

    def cosine_similarity(self, x: TensorLike, y: TensorLike) -> Tensor:
        """
        Compute the cosine similarity between two vectors.

        Args:
            x: First vector
            y: Second vector

        Returns:
            Cosine similarity between -1 and 1
        """
        x_tensor = convert_to_tensor(x)
        y_tensor = convert_to_tensor(y)

        dot_product = torch.sum(torch.multiply(x_tensor, y_tensor))
        norm_x = torch.sqrt(torch.sum(torch.square(x_tensor)))
        norm_y = torch.sqrt(torch.sum(torch.square(y_tensor)))
        return torch.divide(dot_product, torch.add(torch.multiply(norm_x, norm_y), convert_to_tensor(1e-8)))

    def exponential_decay(self, initial_value: TensorLike, decay_rate: TensorLike, time_step: TensorLike) -> Tensor:
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
        return torch.multiply(
            initial_tensor,
            torch.exp(torch.multiply(torch.negative(decay_tensor), time_tensor))
        )

    def gaussian(self, x: TensorLike, mu: TensorLike = 0.0, sigma: TensorLike = 1.0) -> Tensor:
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
        half = convert_to_tensor(0.5)
        two = convert_to_tensor(2.0)
        pi_tensor = convert_to_tensor(pi)

        exponent = torch.multiply(
            torch.negative(half),
            torch.square(torch.divide(torch.subtract(x_tensor, mu_tensor), sigma_tensor))
        )
        denominator = torch.multiply(
            sigma_tensor,
            torch.multiply(torch.sqrt(two), torch.sqrt(pi_tensor))
        )
        return torch.divide(torch.exp(exponent), denominator)

    def compute_energy_stability(self, wave: TensorLike, window_size: int = 100) -> float:
        """
        Compute the energy stability of a wave signal.

        Args:
            wave: Wave signal
            window_size: Window size for stability computation

        Returns:
            Energy stability metric between 0 and 1
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
            energy = torch.sum(torch.square(window))
            energies.append(energy.item())

        if len(energies) <= 1:
            return 1.0

        energies_tensor = convert_to_tensor(energies)
        energy_mean = torch.mean(energies_tensor)
        
        if energy_mean == 0:
            return 1.0

        energy_var = torch.var(energies_tensor)
        stability = torch.divide(convert_to_tensor(1.0), 
                               torch.add(convert_to_tensor(1.0), 
                                       torch.divide(energy_var, energy_mean)))

        return float(stability.item())

    def compute_interference_strength(self, wave1: TensorLike, wave2: TensorLike) -> float:
        """
        Compute the interference strength between two wave signals.

        Args:
            wave1: First wave signal
            wave2: Second wave signal

        Returns:
            Interference strength metric between 0 and 1
        """
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute correlation
        wave1_mean = torch.mean(wave1_tensor)
        wave2_mean = torch.mean(wave2_tensor)
        wave1_centered = torch.subtract(wave1_tensor, wave1_mean)
        wave2_centered = torch.subtract(wave2_tensor, wave2_mean)

        numerator = torch.sum(torch.multiply(wave1_centered, wave2_centered))
        denominator = torch.multiply(
            torch.sqrt(torch.sum(torch.square(wave1_centered))),
            torch.sqrt(torch.sum(torch.square(wave2_centered)))
        )
        denominator = torch.add(denominator, convert_to_tensor(1e-8))
        correlation = torch.divide(numerator, denominator)

        # Compute phase difference using FFT
        fft1 = torch.fft.fft(wave1_tensor)
        fft2 = torch.fft.fft(wave2_tensor)
        phase1 = torch.angle(fft1)
        phase2 = torch.angle(fft2)
        phase_diff = torch.abs(torch.subtract(phase1, phase2))
        mean_phase_diff = torch.mean(phase_diff)

        # Normalize phase difference to [0, 1]
        pi_tensor = convert_to_tensor(pi)
        normalized_phase_diff = torch.divide(mean_phase_diff, pi_tensor)

        # Compute interference strength
        interference_strength = torch.multiply(
            correlation, 
            torch.subtract(convert_to_tensor(1.0), normalized_phase_diff)
        )

        return float(interference_strength.item())

    def compute_phase_coherence(self, wave1: TensorLike, wave2: TensorLike, 
                              freq_range: Optional[Tuple[float, float]] = None) -> float:
        """
        Compute the phase coherence between two wave signals.

        Args:
            wave1: First wave signal
            wave2: Second wave signal
            freq_range: Optional frequency range to consider (min_freq, max_freq)

        Returns:
            Phase coherence value between 0 and 1
        """
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute FFT
        fft1 = torch.fft.fft(wave1_tensor)
        fft2 = torch.fft.fft(wave2_tensor)

        # Get phases
        phase1 = torch.angle(fft1)
        phase2 = torch.angle(fft2)

        # Get frequencies
        freqs = torch.fft.fftfreq(len(wave1_tensor))

        # Apply frequency range filter if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            min_freq_tensor = convert_to_tensor(min_freq)
            max_freq_tensor = convert_to_tensor(max_freq)
            freq_mask = torch.logical_and(
                torch.greater_equal(torch.abs(freqs), min_freq_tensor),
                torch.less_equal(torch.abs(freqs), max_freq_tensor)
            )
            phase1 = torch.where(freq_mask, phase1, torch.zeros_like(phase1))
            phase2 = torch.where(freq_mask, phase2, torch.zeros_like(phase2))

        # Compute phase difference
        phase_diff = torch.subtract(phase1, phase2)

        # Use Euler's formula for complex phase calculation
        complex_real = torch.cos(phase_diff)
        complex_imag = torch.sin(phase_diff)
        coherence = torch.sqrt(torch.add(
            torch.square(torch.mean(complex_real)),
            torch.square(torch.mean(complex_imag))
        ))

        return float(coherence.item())

    def partial_interference(self, wave1: TensorLike, wave2: TensorLike, window_size: int = 100) -> Tensor:
        """
        Compute the partial interference between two wave signals over sliding windows.

        Args:
            wave1: First wave signal
            wave2: Second wave signal
            window_size: Size of the sliding window

        Returns:
            Array of interference values for each window
        """
        wave1_tensor = convert_to_tensor(wave1)
        wave2_tensor = convert_to_tensor(wave2)

        # Ensure waves are the same length
        min_length = min(wave1_tensor.shape[0], wave2_tensor.shape[0])
        wave1_tensor = wave1_tensor[:min_length]
        wave2_tensor = wave2_tensor[:min_length]

        # Compute number of windows
        num_windows = min_length - window_size + 1
        interference = []

        # Compute interference for each window
        for i in range(num_windows):
            window1 = wave1_tensor[i:i + window_size]
            window2 = wave2_tensor[i:i + window_size]

            # Compute correlation
            window1_mean = torch.mean(window1)
            window2_mean = torch.mean(window2)
            window1_centered = torch.subtract(window1, window1_mean)
            window2_centered = torch.subtract(window2, window2_mean)

            correlation = torch.divide(
                torch.sum(torch.multiply(window1_centered, window2_centered)),
                torch.add(torch.multiply(
                    torch.sqrt(torch.sum(torch.square(window1_centered))),
                    torch.sqrt(torch.sum(torch.square(window2_centered)))
                ), convert_to_tensor(1e-8))
            )

            # Compute FFT for this window
            fft1 = torch.fft.fft(window1)
            fft2 = torch.fft.fft(window2)

            # Get phases
            phase1 = torch.angle(fft1)
            phase2 = torch.angle(fft2)
            phase_diff = torch.abs(torch.subtract(phase1, phase2))
            mean_phase_diff = torch.mean(phase_diff)

            # Normalize phase difference to [0, 1]
            pi_tensor = convert_to_tensor(pi)
            normalized_phase_diff = torch.divide(mean_phase_diff, pi_tensor)

            # Compute interference strength
            interference.append(torch.multiply(
                correlation, 
                torch.subtract(convert_to_tensor(1.0), normalized_phase_diff)
            ))

        return convert_to_tensor(interference)

    def fft(self, a: TensorLike, n: Optional[int] = None, axis: int = -1) -> Tensor:
        """
        One dimensional discrete Fourier Transform.

        Args:
            a: Input array
            n: Length of the transformed axis
            axis: Axis over which to compute the FFT

        Returns:
            The transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.fft(a_tensor, n=n, dim=axis)

    def ifft(self, a: TensorLike, n: Optional[int] = None, axis: int = -1) -> Tensor:
        """
        One dimensional inverse discrete Fourier Transform.

        Args:
            a: Input array
            n: Length of the transformed axis
            axis: Axis over which to compute the IFFT

        Returns:
            The inverse transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.ifft(a_tensor, n=n, dim=axis)

    def fft2(self, a: TensorLike, s: Optional[Tuple[int, int]] = None, 
             axes: Tuple[int, int] = (-2, -1)) -> Tensor:
        """
        Two dimensional discrete Fourier Transform.

        Args:
            a: Input array
            s: Shape of the transformed axes
            axes: Axes over which to compute the FFT2

        Returns:
            The transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.fft2(a_tensor, s=s, dim=axes)

    def ifft2(self, a: TensorLike, s: Optional[Tuple[int, int]] = None, 
              axes: Tuple[int, int] = (-2, -1)) -> Tensor:
        """
        Two dimensional inverse discrete Fourier Transform.

        Args:
            a: Input array
            s: Shape of the transformed axes
            axes: Axes over which to compute the IFFT2

        Returns:
            The inverse transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.ifft2(a_tensor, s=s, dim=axes)

    def fftn(self, a: TensorLike, s: Optional[Sequence[int]] = None, 
             axes: Optional[Sequence[int]] = None) -> Tensor:
        """
        N-dimensional discrete Fourier Transform.

        Args:
            a: Input array
            s: Shape of the transformed axes
            axes: Axes over which to compute the FFTN

        Returns:
            The transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.fftn(a_tensor, s=s, dim=axes)

    def ifftn(self, a: TensorLike, s: Optional[Sequence[int]] = None, 
              axes: Optional[Sequence[int]] = None) -> Tensor:
        """
        N-dimensional inverse discrete Fourier Transform.

        Args:
            a: Input array
            s: Shape of the transformed axes
            axes: Axes over which to compute the IFFTN

        Returns:
            The inverse transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.ifftn(a_tensor, s=s, dim=axes)

    def rfft(self, a: TensorLike, n: Optional[int] = None, axis: int = -1) -> Tensor:
        """
        One dimensional real discrete Fourier Transform.

        Args:
            a: Input array (real)
            n: Length of the transformed axis
            axis: Axis over which to compute the RFFT

        Returns:
            The transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.rfft(a_tensor, n=n, dim=axis)

    def irfft(self, a: TensorLike, n: Optional[int] = None, axis: int = -1) -> Tensor:
        """
        One dimensional inverse real discrete Fourier Transform.

        Args:
            a: Input array
            n: Length of the transformed axis
            axis: Axis over which to compute the IRFFT

        Returns:
            The inverse transformed array (real)
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.irfft(a_tensor, n=n, dim=axis)

    def rfft2(self, a: TensorLike, s: Optional[Tuple[int, int]] = None, 
              axes: Tuple[int, int] = (-2, -1)) -> Tensor:
        """
        Two dimensional real discrete Fourier Transform.

        Args:
            a: Input array (real)
            s: Shape of the transformed axes
            axes: Axes over which to compute the RFFT2

        Returns:
            The transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.rfft2(a_tensor, s=s, dim=axes)

    def irfft2(self, a: TensorLike, s: Optional[Tuple[int, int]] = None, 
               axes: Tuple[int, int] = (-2, -1)) -> Tensor:
        """
        Two dimensional inverse real discrete Fourier Transform.

        Args:
            a: Input array
            s: Shape of the transformed axes
            axes: Axes over which to compute the IRFFT2

        Returns:
            The inverse transformed array (real)
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.irfft2(a_tensor, s=s, dim=axes)

    def rfftn(self, a: TensorLike, s: Optional[Sequence[int]] = None, 
              axes: Optional[Sequence[int]] = None) -> Tensor:
        """
        N-dimensional real discrete Fourier Transform.

        Args:
            a: Input array (real)
            s: Shape of the transformed axes
            axes: Axes over which to compute the RFFTN

        Returns:
            The transformed array
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.rfftn(a_tensor, s=s, dim=axes)

    def irfftn(self, a: TensorLike, s: Optional[Sequence[int]] = None, 
               axes: Optional[Sequence[int]] = None) -> Tensor:
        """
        N-dimensional inverse real discrete Fourier Transform.

        Args:
            a: Input array
            s: Shape of the transformed axes
            axes: Axes over which to compute the IRFFTN

        Returns:
            The inverse transformed array (real)
        """
        a_tensor = convert_to_tensor(a)
        return torch.fft.irfftn(a_tensor, s=s, dim=axes)
