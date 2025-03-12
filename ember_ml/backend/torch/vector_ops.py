"""
PyTorch implementation of vector operations.

This module provides PyTorch implementations of vector operations.
"""

import torch
from typing import Optional, Tuple, Any

from ember_ml.ops.interfaces.vector_ops import VectorOps
from ember_ml.backend.torch.math_ops import TorchMathOps
from ember_ml.backend.torch.tensor_ops import convert_to_tensor


class TorchVectorOps(VectorOps):
    """PyTorch implementation of vector operations."""
    
    def normalize_vector(self, vector: Any) -> Any:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        
        norm = torch.norm(vector, p=2)
        if norm > 0:
            return vector / norm
        return vector
    
    def compute_energy_stability(self, wave: Any, window_size: int = 100) -> float:
        """
        Compute the energy stability of a wave signal.
        
        Args:
            wave: Wave signal
            window_size: Window size for stability computation
            
        Returns:
            Energy stability metric
        """
        if not isinstance(wave, torch.Tensor):
            wave = torch.tensor(wave, dtype=torch.float32)
            
        if len(wave) < window_size:
            return 1.0  # Perfectly stable for short signals
            
        # Compute energy in windows
        num_windows = len(wave) // window_size
        energies = []
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = wave[start:end]
            energy = torch.sum(torch.square(window))
            energies.append(energy.item())
        
        # Compute stability as inverse of energy variance
        if len(energies) <= 1:
            return 1.0
            
        energies_tensor = torch.tensor(energies)
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
        # Convert to PyTorch tensors if needed
        if not isinstance(wave1, torch.Tensor):
            wave1 = torch.tensor(wave1, dtype=torch.float32)
        if not isinstance(wave2, torch.Tensor):
            wave2 = torch.tensor(wave2, dtype=torch.float32)
            
        # Ensure waves are the same length
        min_length = min(len(wave1), len(wave2))
        wave1 = wave1[:min_length]
        wave2 = wave2[:min_length]
        
        # Compute correlation
        # PyTorch doesn't have a direct corrcoef function, so we compute it manually
        wave1_centered = wave1 - torch.mean(wave1)
        wave2_centered = wave2 - torch.mean(wave2)
        correlation = torch.sum(wave1_centered * wave2_centered) / (
            torch.sqrt(torch.sum(wave1_centered ** 2)) * torch.sqrt(torch.sum(wave2_centered ** 2)) + 1e-8
        )
        
        # Compute phase difference using PyTorch's FFT
        fft1 = torch.fft.fft(wave1.float())
        fft2 = torch.fft.fft(wave2.float())
        phase1 = torch.angle(fft1)
        phase2 = torch.angle(fft2)
        phase_diff = torch.abs(phase1 - phase2)
        mean_phase_diff = torch.mean(phase_diff)
        
        # Normalize phase difference to [0, 1]
        normalized_phase_diff = mean_phase_diff / TorchMathOps.pi
        
        # Compute interference strength
        # High correlation and low phase difference = constructive interference
        # Low correlation and high phase difference = destructive interference
        interference_strength = correlation.item() * (1.0 - normalized_phase_diff)
        
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
        # Convert to PyTorch tensors if needed
        if not isinstance(wave1, torch.Tensor):
            wave1 = torch.tensor(wave1, dtype=torch.float32)
        if not isinstance(wave2, torch.Tensor):
            wave2 = torch.tensor(wave2, dtype=torch.float32)
            
        # Ensure waves are the same length
        min_length = min(len(wave1), len(wave2))
        wave1 = wave1[:min_length]
        wave2 = wave2[:min_length]
        
        # Compute FFT using PyTorch
        fft1 = torch.fft.fft(wave1.float())
        fft2 = torch.fft.fft(wave2.float())
        
        # Get phases
        phase1 = torch.angle(fft1)
        phase2 = torch.angle(fft2)
        
        # Get frequencies
        freqs = torch.fft.fftfreq(len(wave1))
        
        # Apply frequency range filter if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            freq_mask = (torch.abs(freqs) >= min_freq) & (torch.abs(freqs) <= max_freq)
            phase1 = phase1[freq_mask]
            phase2 = phase2[freq_mask]
        
        # Compute phase difference
        phase_diff = phase1 - phase2
        
        # Compute phase coherence using circular statistics
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
        # Convert to PyTorch tensors if needed
        if not isinstance(wave1, torch.Tensor):
            wave1 = torch.tensor(wave1, dtype=torch.float32)
        if not isinstance(wave2, torch.Tensor):
            wave2 = torch.tensor(wave2, dtype=torch.float32)
            
        # Ensure waves are the same length
        min_length = min(len(wave1), len(wave2))
        wave1 = wave1[:min_length]
        wave2 = wave2[:min_length]
        
        # Compute number of windows
        num_windows = min_length - window_size + 1
        
        # Initialize result array
        interference = torch.zeros(num_windows, dtype=torch.float32)
        
        # Compute interference for each window
        for i in range(num_windows):
            window1 = wave1[i:i+window_size]
            window2 = wave2[i:i+window_size]
            
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
            normalized_phase_diff = mean_phase_diff / TorchMathOps.pi
            
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
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        return torch.sqrt(torch.sum(torch.square(x - y)))
    
    def cosine_similarity(self, x: Any, y: Any) -> Any:
        """
        Compute the cosine similarity between two vectors.
        
        Args:
            x: First vector
            y: Second vector
            
        Returns:
            Cosine similarity
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        dot_product = torch.sum(x * y)
        norm_x = torch.sqrt(torch.sum(torch.square(x)))
        norm_y = torch.sqrt(torch.sum(torch.square(y)))
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
        if not isinstance(initial_value, torch.Tensor):
            initial_value = torch.tensor(initial_value, dtype=torch.float32)
        if not isinstance(decay_rate, torch.Tensor):
            decay_rate = torch.tensor(decay_rate, dtype=torch.float32)
        if not isinstance(time_step, torch.Tensor):
            time_step = torch.tensor(time_step, dtype=torch.float32)
            
        return initial_value * torch.exp(-decay_rate * time_step)
    
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
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
            
        return torch.exp(-0.5 * torch.square((x - mu) / sigma)) / (sigma * torch.sqrt(torch.tensor(2.0)) * torch.sqrt(TorchMathOps.pi))