"""Test diag and diagonal functions."""

import numpy as np
from ember_ml.ops.linearalg import diag, diagonal
from ember_ml.nn import tensor
from ember_ml.backend import set_backend

def test_diag():
    """Test diag function."""
    # Test with 1D input
    x = tensor.convert_to_tensor([1, 2, 3])
    result = diag(x)
    print("Diag with 1D input:")
    print(result)
    
    # Test with 2D input
    x = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = diag(x)
    print("\nDiag with 2D input:")
    print(result)
    
    # Test with offset
    result = diag(x, k=1)
    print("\nDiag with 2D input and offset 1:")
    print(result)

def test_diagonal():
    """Test diagonal function."""
    # Create a 3D tensor
    x = tensor.convert_to_tensor(np.arange(24).reshape(2, 3, 4))
    
    # Test diagonal with default parameters
    result = diagonal(x)
    print("\nDiagonal with default parameters:")
    print(result)
    
    # Test diagonal with offset
    result = diagonal(x, offset=1)
    print("\nDiagonal with offset 1:")
    print(result)
    
    # Test diagonal with different axes
    result = diagonal(x, axis1=0, axis2=2)
    print("\nDiagonal with axis1=0, axis2=2:")
    print(result)

if __name__ == "__main__":
    # Test with NumPy backend
    set_backend('mlx')
    print("Testing with NumPy backend:")
    test_diag()
    test_diagonal()
    
    # Test with MLX backend if available
    try:
        set_backend('mlx')
        print("\nTesting with MLX backend:")
        test_diag()
        test_diagonal()
    except (ImportError, ValueError):
        print("\nMLX backend not available")