"""
Test file for EmberTensor backend operations.
"""

from ember_ml import ops
from ember_ml.backend import set_backend, get_backend

def test_ember_tensor_ops():
    """Test EmberTensor operations using the ops module."""
    # Set the backend to ember
    set_backend('ember')
    
    # Create tensors
    a = ops.convert_to_tensor([1, 2, 3])
    b = ops.convert_to_tensor([4, 5, 6])
    
    print(f"Backend: {get_backend()}")
    print(f"a = {a}")
    print(f"b = {b}")
    
    # Test ops module functions
    print(f"ops.add(a, b) = {ops.add(a, b)}")
    print(f"ops.subtract(a, b) = {ops.subtract(a, b)}")
    print(f"ops.multiply(a, b) = {ops.multiply(a, b)}")
    print(f"ops.divide(a, b) = {ops.divide(a, b)}")
    
    # Test using Python operators (which should use the operator methods internally)
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_ember_tensor_ops()