"""
Random number generation utilities.

This module provides utilities for random number generation,
including seeding the random number generator.
"""

from ember_ml import ops

def seed(seed_value: int) -> None:
    """
    Seed the random number generators.
    
    This function seeds the random number generators in the current backend.
    
    Args:
        seed_value: Seed value for the random number generators
    """
    ops.set_seed(seed_value)

# Alias for backward compatibility
set_random_seed = seed