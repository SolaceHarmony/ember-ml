import numpy as np
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.rnn import StrideAwareCell

# Create a cell
cell = StrideAwareCell(
    input_size=10,
    hidden_size=20,
    stride_length=3,
    time_scale_factor=1.5
)

# Print the cell's parameters
print("Cell parameters:")
for name, param in cell._parameters.items():
    print(f"- {name}: {param}")

# Try to access parameters directly
print("\nAccessing parameters directly:")
print(f"- input_kernel: {cell.input_kernel}")
print(f"- hidden_kernel: {cell.hidden_kernel}")
print(f"- tau: {cell.tau}")

# Create input
batch_size = 2
inputs = tensor.random_normal((batch_size, 10))

# Initialize state
state = tensor.zeros((batch_size, 20))

# Forward pass
output, new_state = cell(inputs, state)

print(f"\nInput shape: {tensor.shape(inputs)}")
print(f"Output shape: {tensor.shape(output)}")
print(f"State shape: {tensor.shape(new_state)}")