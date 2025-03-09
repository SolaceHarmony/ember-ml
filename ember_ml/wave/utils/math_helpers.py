"""
Math helper functions for wave processing.

This module re-exports math helper functions from emberharmony.utils.math_helpers
for use in wave processing, and adds wave-specific math functions.
"""

import numpy as np
import torch

# Re-export math helpers from utils module
from ember_ml.utils.math_helpers import (
    sigmoid,
    tanh,
    relu,
    leaky_relu,
    softmax,
    normalize,
    standardize,
    euclidean_distance,
    cosine_similarity,
    exponential_decay,
    gaussian
)

def normalize_vector(vector: np.ndarray) -> np.ndarray:
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

def normalize_vector_torch(vector: torch.Tensor) -> torch.Tensor:
    """
    Normalize a PyTorch tensor to unit length.
    
    Args:
        vector: Input tensor
        
    Returns:
        Normalized tensor
    """
    norm = torch.norm(vector, p=2, dim=-1, keepdim=True)
    return vector / (norm + 1e-8)

def compute_energy_stability(wave: np.ndarray, window_size: int = 100) -> float:
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
        energy = np.sum(window ** 2)
        energies.append(energy)
    
    # Compute stability as inverse of energy variance
    if len(energies) <= 1:
        return 1.0
        
    energy_mean = np.mean(energies)
    if energy_mean == 0:
        return 1.0
        
    energy_var = np.var(energies)
    stability = 1.0 / (1.0 + energy_var / energy_mean)
    
    return stability

def compute_interference_strength(wave1: np.ndarray, wave2: np.ndarray) -> float:
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
    
    return interference_strength

def compute_phase_coherence(wave1: np.ndarray, wave2: np.ndarray, freq_range: tuple = None) -> float:
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
    
    return coherence

def partial_interference(wave1: np.ndarray, wave2: np.ndarray, window_size: int = 100) -> np.ndarray:
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

__all__ = [
    'sigmoid',
    'tanh',
    'relu',
    'leaky_relu',
    'softmax',
    'normalize',
    'standardize',
    'euclidean_distance',
    'cosine_similarity',
    'exponential_decay',
    'gaussian',
    'normalize_vector',
    'normalize_vector_torch',
    'compute_energy_stability',
    'compute_interference_strength',
    'compute_phase_coherence',
    'partial_interference'
]