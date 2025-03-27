"""Orthogonal initializer."""

from typing import Sequence
from ember_ml.core import backend as K

def orthogonal(shape: Sequence[int], gain: float = 1.0) -> K.TensorLike:
    """
    Generate a (semi-)orthogonal matrix or tensor.

    If the shape has more than two dimensions, the matrix is generated for the first
    two dimensions and then reshaped.

    Args:
        shape: Shape of the tensor to initialize. Must have at least 2 dimensions.
        gain: Multiplicative factor to apply to the orthogonal matrix.

    Returns:
        An initialized tensor.

    Raises:
        ValueError: If shape has less than 2 dimensions.
    """
    # Dispatch to the backend-specific implementation
    return K.backend_factory().initializers.orthogonal(shape, gain=gain)