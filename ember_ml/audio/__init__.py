"""Audio processing and transformation module.

This module provides backend-agnostic implementations of audio processing
operations, including harmonic wave generation and variable quantization.

Components:
    HarmonicWaveDemo: Demonstration of harmonic wave generation
        - Frequency synthesis
        - Wave superposition
        - Harmonic series generation
        
    variablequantization: Advanced audio quantization tools
        - Adaptive bit depth adjustment
        - Dynamic range optimization
        - Non-linear quantization schemes

All implementations use the ops abstraction layer for computations and
maintain strict backend independence.
"""

from ember_ml.audio.HarmonicWaveDemo import *
from ember_ml.audio.variablequantization import *

__all__ = [
    'HarmonicWaveDemo',
    'variablequantization',
]
