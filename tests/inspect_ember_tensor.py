"""
Script to inspect an EmberTensor.
"""

from ember_ml import tensor
import numpy as np

# Create an EmberTensor
data = np.array([[1, 2, 3], [4, 5, 6]])
tensor = tensor(data)

# Inspect the tensor
print("Type:", type(tensor))
print("Dir:", dir(tensor))
print("_tensor attribute:", tensor._tensor)
print("Type of _tensor:", type(tensor._tensor))