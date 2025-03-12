"""
NumPy implementation of vector operations.

This module provides NumPy implementations of vector operations.
"""

import numpy as np
from typing import Optional, Tuple, Any

from ember_ml.ops.interfaces.vector_ops import VectorOps


class NumpyVectorOps(VectorOps):
    """NumPy implementation of vector operations."""
    
    def normalize_vector(self, vector: Any) -> Any:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
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
        if len(wave) < window_size:
            return 1.0  # Perfectly stable for short signals
            
        # Compute energy in windows
        num_windows = len(wave) // window_size
        energies = []
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = wave[start:end]
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
        # Ensure waves are the same length
        min_length = min(len(wave1), len(wave2))
        wave1 = wave1[:min_length]
        wave2 = wave2[:min_length]
        
        # Compute correlation
        correlation = np.corrcoef(wave1, wave2)[0, 1]
        
        # Compute phase difference
        fft1 = np.fft.fft(wave1)
        fft2 = np.fft.fft(wave2)
        phase1 = np.angle(fft1)
        phase2 = np.angle(fft2)
        phase_diff = np.abs(phase1 - phase2)
        mean_phase_diff = np.mean(phase_diff)
        
        # Normalize phase difference to [0, 1]
        normalized_phase_diff = mean_phase_diff / np.pi
        
        # Compute interference strength
        # High correlation and low phase difference = constructive interference
        # Low correlation and high phase difference = destructive interference
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
        # Ensure waves are the same length
        min_length = min(len(wave1), len(wave2))
        wave1 = wave1[:min_length]
        wave2 = wave2[:min_length]
        
        # Compute FFT
        fft1 = np.fft.fft(wave1)
        fft2 = np.fft.fft(wave2)
        
        # Get phases
        phase1 = np.angle(fft1)
        phase2 = np.angle(fft2)
        
        # Get frequencies
        freqs = np.fft.fftfreq(len(wave1))
        
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
        # Ensure waves are the same length
        min_length = min(len(wave1), len(wave2))
        wave1 = wave1[:min_length]
        wave2 = wave2[:min_length]
        
        # Compute number of windows
        num_windows = min_length - window_size + 1
        
        # Initialize result array
        interference = np.zeros(num_windows)
        
        # Compute interference for each window
        for i in range(num_windows):
            window1 = wave1[i:i+window_size]
            window2 = wave2[i:i+window_size]
            
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
            normalized_phase_diff = mean_phase_diff / np.pi
            
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
        return np.sqrt(np.sum(np.square(x - y)))
    
    def cosine_similarity(self, x: Any, y: Any) -> Any:
        """
        Compute the cosine similarity between two vectors.
        
        Args:
            x: First vector
            y: Second vector
            
        Returns:
            Cosine similarity
        """
        dot_product = np.sum(x * y)
        norm_x = np.sqrt(np.sum(np.square(x)))
        norm_y = np.sqrt(np.sum(np.square(y)))
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
        return initial_value * np.exp(-decay_rate * time_step)
    
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
        return np.exp(-0.5 * np.square((x - mu) / sigma)) / (sigma * np.sqrt(2.0 * np.pi))