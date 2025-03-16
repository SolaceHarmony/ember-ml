"""
Test script for RBM save/load functionality.
"""

from ember_ml import ops
from ember_ml.models.rbm import RBMModule, save_rbm, load_rbm
import tempfile
import os

def main():
    # Set backend to numpy
    ops.set_backend('numpy')
    print(f'Current backend: {ops.get_backend()}')
    
    # Create an RBM
    rbm = RBMModule(n_visible=10, n_hidden=5)
    print(f'Original RBM type: {type(rbm)}')
    print(f'Original RBM has weights: {hasattr(rbm, "weights")}')
    
    # Save the RBM
    temp_file = os.path.join(tempfile.gettempdir(), 'rbm_test.npy')
    save_rbm(rbm, temp_file)
    print(f'Saved RBM to {temp_file}')
    
    # Load the RBM
    try:
        loaded_rbm = load_rbm(temp_file)
        print(f'Loaded RBM type: {type(loaded_rbm)}')
        print(f'Loaded RBM has weights: {hasattr(loaded_rbm, "weights")}')
    except Exception as e:
        print(f'Error loading RBM: {e}')
    
    # Try to load with numpy directly to see what's stored
    try:
        import numpy as np
        loaded_data = np.load(temp_file, allow_pickle=True)
        print(f'Numpy loaded data type: {type(loaded_data)}')
        print(f'Is instance of RBMModule: {isinstance(loaded_data, RBMModule)}')
    except Exception as e:
        print(f'Error loading with numpy: {e}')

if __name__ == '__main__':
    main()