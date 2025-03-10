"""
Restricted Boltzmann Machine (RBM) module.

This module provides Restricted Boltzmann Machine (RBM) models for the ember_ml library.
"""

from ember_ml.models.rbm.rbm import RestrictedBoltzmannMachine
from ember_ml.models.rbm.rbm import train_rbm, reconstruct_with_rbm

__all__ = [
    'RestrictedBoltzmannMachine',
    'train_rbm',
    'reconstruct_with_rbm',
]
