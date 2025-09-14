"""
Example script to test the new Ember ML API.
"""

import ember_ml as em
# No direct backend imports allowed in frontend code

def test_tensor_creation():
    """if __name__ == "__main__":
    print("Testing new Ember ML API...")
    print("==========================\n")
    
    # Test backend purity first
    test_backend_purity()
    
    # Test tensor creation
    test_tensor_creation()
    
    # Test math operations
    test_math_operations()
    
    # Test linear algebra operations
    test_linalg_operations()
    
    # Test statistics operations
    test_stats_operations()
    
    # Test activation functions
    test_activation_functions()
    
    print("All tests completed!") functions."""
    print("Testing tensor creation functions...")
    
    # Create arrays - pure backend-agnostic code
    a = em.array([1, 2, 3])
    print(f"array([1, 2, 3]) = shape: {em.shape(a)}")
    
    # Create zeros
    z = em.zeros((2, 3))
    print(f"zeros((2, 3)) = shape: {em.shape(z)}")
    
    # Create ones
    o = em.ones((2, 2))
    print(f"ones((2, 2)) = shape: {em.shape(o)}")
    
    # Create eye
    e = em.eye(3)
    print(f"eye(3) = shape: {em.shape(e)}")
    
    print("Tensor creation tests completed.\n")
    
def test_math_operations():
    """Test basic math operations."""
    print("Testing math operations...")
    
    # Create tensors - pure backend-agnostic code
    a = em.array([[1, 2], [3, 4]])
    b = em.array([[5, 6], [7, 8]])
    
    # Add
    c = em.add(a, b)
    print(f"add(a, b) shape: {em.shape(c)}")
    
    # Subtract
    d = em.subtract(a, b)
    print(f"subtract(a, b) shape: {em.shape(d)}")
    
    # Multiply
    e = em.multiply(a, b)
    print(f"multiply(a, b) shape: {em.shape(e)}")
    
    # Matrix multiplication
    f = em.matmul(a, b)
    print(f"matmul(a, b) shape: {em.shape(f)}")
    
    # Reshape
    g = em.reshape(a, (4,))
    print(f"reshape(a, (4,)) shape: {em.shape(g)}")
    
    print("Math operations tests completed.\n")

def test_linalg_operations():
    """Test linear algebra operations."""
    print("Testing linear algebra operations...")
    
    # Create a matrix - pure backend-agnostic code
    a = em.array([[1, 2], [3, 4]])
    
    # SVD
    try:
        u, s, vh = em.linalg.svd(a)
        print(f"linalg.svd(a) shapes: u={em.shape(u)}, s={em.shape(s)}, vh={em.shape(vh)}")
    except (AttributeError, ImportError):
        print("SVD not registered or available.")
    
    # QR decomposition
    try:
        q, r = em.linalg.qr(a)
        print(f"linalg.qr(a) shapes: q={em.shape(q)}, r={em.shape(r)}")
    except (AttributeError, ImportError):
        print("QR decomposition not registered or available.")
    
    print("Linear algebra tests completed.\n")
    
def test_stats_operations():
    """Test statistics operations."""
    print("Testing statistics operations...")
    
    # Create a tensor - pure backend-agnostic code
    a = em.array([1, 2, 3, 4, 5])
    
    # Mean
    try:
        m = em.stats.mean(a)
        print(f"stats.mean(a) shape: {em.shape(m)}")
    except (AttributeError, ImportError):
        print("Mean not registered or available.")
    
    # Standard deviation
    try:
        s = em.stats.std(a)
        print(f"stats.std(a) shape: {em.shape(s)}")
    except (AttributeError, ImportError):
        print("Standard deviation not registered or available.")
    
    print("Statistics tests completed.\n")

def test_activation_functions():
    """Test activation functions."""
    print("Testing activation functions...")
    
    # Create a tensor - pure backend-agnostic code
    a = em.array([-2, -1, 0, 1, 2])
    
    # ReLU
    try:
        r = em.activations.relu(a)
        print(f"activations.relu(a) shape: {em.shape(r)}")
    except (AttributeError, ImportError):
        print("ReLU not registered or available.")
    
    # Sigmoid
    try:
        s = em.activations.sigmoid(a)
        print(f"activations.sigmoid(a) shape: {em.shape(s)}")
    except (AttributeError, ImportError):
        print("Sigmoid not registered or available.")
    
    print("Activation functions tests completed.\n")

def test_backend_purity():
    """Test that no backend-specific types are exposed."""
    print("Testing backend purity...")
    
    # Create a tensor
    a = em.array([1, 2, 3])
    
    # Check tensor type
    tensor_type = type(a).__name__
    print(f"Tensor type: {tensor_type}")
    
    # Check module name
    module_name = type(a).__module__
    print(f"Module name: {module_name}")
    
    # Verify no direct NumPy exposure
    numpy_exposed = 'numpy' in module_name or tensor_type == 'ndarray'
    torch_exposed = 'torch' in module_name or tensor_type == 'Tensor'
    mlx_exposed = 'mlx' in module_name or tensor_type == 'array'
    
    if numpy_exposed:
        print("WARNING: NumPy appears to be directly exposed in the frontend!")
    else:
        print("✅ No direct NumPy exposure")
        
    if torch_exposed:
        print("WARNING: PyTorch appears to be directly exposed in the frontend!")
    else:
        print("✅ No direct PyTorch exposure")
        
    if mlx_exposed:
        print("WARNING: MLX appears to be directly exposed in the frontend!")
    else:
        print("✅ No direct MLX exposure")
    
    print("Backend purity tests completed.\n")

if __name__ == "__main__":
    print("Testing new Ember ML API...")
    print("==========================\n")
    
    # Test tensor creation
    test_tensor_creation()
    
    # Test math operations
    test_math_operations()
    
    # Test linear algebra operations
    test_linalg_operations()
    
    # Test statistics operations
    test_stats_operations()
    
    # Test activation functions
    test_activation_functions()
    
    # Test backend purity
    test_backend_purity()
    
    print("All tests completed!")
