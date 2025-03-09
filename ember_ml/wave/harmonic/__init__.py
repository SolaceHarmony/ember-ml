"""
Harmonic embeddings module.

This module provides implementations of harmonic embeddings,
including wave generators and embedding utilities.
"""

from ember_ml.wave.harmonic.embedding_utils import *
from ember_ml.wave.harmonic.wave_generator import *
from ember_ml.wave.harmonic.visualization import *
from ember_ml.wave.harmonic.training import *

# Import directly from the parent module
import sys
import importlib
if 'emberharmony.wave.harmonic' in sys.modules:
    # Get the parent module
    parent_module = importlib.import_module('emberharmony.wave')
    
    # Import the classes from the parent module
    HarmonicProcessor = getattr(parent_module, 'HarmonicProcessor', None)
    FrequencyAnalyzer = getattr(parent_module, 'FrequencyAnalyzer', None)
    WaveSynthesizer = getattr(parent_module, 'WaveSynthesizer', None)

__all__ = [
    'embedding_utils',
    'wave_generator',
    'visualization',
    'training',
    'HarmonicProcessor',
    'FrequencyAnalyzer',
    'WaveSynthesizer',
]
