"""
Test script to verify the MLX backend fixes.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Set the backend to MLX
    os.environ['ember_ml_OPS'] = 'mlx'
    
    # Import the ops module
    from ember_ml import ops
    
    # Test the pi constant
    print("Testing pi constant...")
    pi_value = ops.pi
    print(f"pi = {pi_value}")
    
    # Test the var function
    print("\nTesting var function...")
    tensor = ops.convert_to_tensor([1, 2, 3, 4, 5])
    variance = ops.var(tensor)
    print(f"Variance of [1, 2, 3, 4, 5] = {variance}")
    
    print("\nAll tests passed successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()