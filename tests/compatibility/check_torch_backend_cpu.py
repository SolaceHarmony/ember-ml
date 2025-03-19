from ember_ml import ops
import matplotlib.pyplot as plt
import numpy as np

# Set backend to torch
ops.set_backend('torch')

# Create an EmberTensor
tensor = ops.ones((10,))
print('Type:', type(tensor))

# Try to convert to CPU and then to NumPy
try:
    # Check if it's a PyTorch tensor
    import torch
    if isinstance(tensor, torch.Tensor):
        # Move to CPU first
        cpu_tensor = tensor.cpu()
        numpy_tensor = cpu_tensor.numpy()
        print('Successfully converted PyTorch tensor to NumPy via CPU')
        
        # Try plotting the NumPy array
        plt.figure()
        plt.plot(numpy_tensor)
        plt.close()
        print('Can plot NumPy array after CPU conversion: True')
    else:
        print('Not a PyTorch tensor')
except Exception as e:
    print('Error during conversion or plotting:', e)