"""
Restricted Boltzmann Machine (RBM) Module

This package provides an implementation of Restricted Boltzmann Machines
using the ember_ml Module system.
"""

from ember_ml.models.rbm.rbm_module import RBMModule
from ember_ml.models.rbm.training import (
    contrastive_divergence_step,
    train_rbm,
    transform_in_chunks,
    save_rbm,
    load_rbm
)

# For backward compatibility with existing code
RestrictedBoltzmannMachine = RBMModule

# Explicitly use imported symbols to satisfy linter

__all__ = [ 
    'RBMModule',
    'RestrictedBoltzmannMachine',
    'contrastive_divergence_step',
    'train_rbm',
    'transform_in_chunks',
    'save_rbm',
    'load_rbm'
]
