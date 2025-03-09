"""
Wave conversion utilities.

This module provides utilities for converting between different wave representations.
"""

import numpy as np
from typing import Union, List, Tuple, Optional

def pcm_to_float(pcm_data: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Convert PCM data to floating point representation.
    
    Args:
        pcm_data: PCM data as numpy array
        dtype: Output data type
        
    Returns:
        Floating point representation
    """
    pcm_data = np.asarray(pcm_data)
    if pcm_data.dtype.kind not in 'iu':
        raise TypeError("PCM data must be integer type")
    
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("Output data type must be floating point")
    
    i = np.iinfo(pcm_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    
    return (pcm_data.astype(dtype) - offset) / abs_max

def float_to_pcm(float_data: np.ndarray, dtype: np.dtype = np.int16) -> np.ndarray:
    """
    Convert floating point data to PCM representation.
    
    Args:
        float_data: Floating point data as numpy array
        dtype: Output PCM data type
        
    Returns:
        PCM representation
    """
    float_data = np.asarray(float_data)
    if float_data.dtype.kind != 'f':
        raise TypeError("Input data must be floating point")
    
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("Output data type must be integer")
    
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    
    return (float_data * abs_max + offset).clip(i.min, i.max).astype(dtype)

def pcm_to_db(pcm_data: np.ndarray, ref: float = 1.0, min_db: float = -80.0) -> np.ndarray:
    """
    Convert PCM data to decibels.
    
    Args:
        pcm_data: PCM data as numpy array
        ref: Reference value
        min_db: Minimum dB value
        
    Returns:
        Decibel representation
    """
    float_data = pcm_to_float(pcm_data)
    power = np.abs(float_data) ** 2
    db = 10 * np.log10(np.maximum(power, 1e-10) / ref)
    return np.maximum(db, min_db)

def db_to_amplitude(db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert decibels to amplitude.
    
    Args:
        db: Decibel value
        
    Returns:
        Amplitude value
    """
    return 10 ** (db / 20)

def amplitude_to_db(amplitude: Union[float, np.ndarray], min_db: float = -80.0) -> Union[float, np.ndarray]:
    """
    Convert amplitude to decibels.
    
    Args:
        amplitude: Amplitude value
        min_db: Minimum dB value
        
    Returns:
        Decibel value
    """
    db = 20 * np.log10(np.maximum(np.abs(amplitude), 1e-10))
    return np.maximum(db, min_db)

def pcm_to_binary(pcm_data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Convert PCM data to binary representation.
    
    Args:
        pcm_data: PCM data as numpy array
        threshold: Threshold for binarization
        
    Returns:
        Binary representation
    """
    float_data = pcm_to_float(pcm_data)
    return (float_data > threshold).astype(np.int8)

def binary_to_pcm(binary_data: np.ndarray, amplitude: float = 1.0, dtype: np.dtype = np.int16) -> np.ndarray:
    """
    Convert binary data to PCM representation.
    
    Args:
        binary_data: Binary data as numpy array
        amplitude: Amplitude of the PCM signal
        dtype: Output PCM data type
        
    Returns:
        PCM representation
    """
    float_data = binary_data.astype(np.float32) * 2 - 1
    float_data *= amplitude
    return float_to_pcm(float_data, dtype)

def pcm_to_phase(pcm_data: np.ndarray) -> np.ndarray:
    """
    Convert PCM data to phase representation.
    
    Args:
        pcm_data: PCM data as numpy array
        
    Returns:
        Phase representation
    """
    float_data = pcm_to_float(pcm_data)
    return np.angle(np.fft.fft(float_data))

def phase_to_pcm(phase_data: np.ndarray, magnitude: Optional[np.ndarray] = None, dtype: np.dtype = np.int16) -> np.ndarray:
    """
    Convert phase data to PCM representation.
    
    Args:
        phase_data: Phase data as numpy array
        magnitude: Magnitude data as numpy array
        dtype: Output PCM data type
        
    Returns:
        PCM representation
    """
    if magnitude is None:
        magnitude = np.ones_like(phase_data)
    
    complex_data = magnitude * np.exp(1j * phase_data)
    float_data = np.real(np.fft.ifft(complex_data))
    return float_to_pcm(float_data, dtype)