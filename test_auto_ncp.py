import numpy as np
from ember_ml import ops
from ember_ml.nn.modules import AutoNCP

# Create an AutoNCP module
auto_ncp = AutoNCP(
    units=15,
    output_size=5,
    sparsity_level=0.5,
    activation="tanh",
    use_bias=True
)

# Print the AutoNCP's parameters
print("AutoNCP parameters:")
for name, param in auto_ncp._parameters.items():
    print(f"- {name}: {param}")

# Try to access parameters directly
print("\nAccessing parameters directly:")
print(f"- kernel: {auto_ncp.kernel}")
print(f"- recurrent_kernel: {auto_ncp.recurrent_kernel}")
print(f"- bias: {auto_ncp.bias}")

# Create input
batch_size = 2
inputs = ops.random_normal((batch_size, 15))

# Forward pass
output = auto_ncp(inputs)

print(f"\nInput shape: {ops.shape(inputs)}")
print(f"Output shape: {ops.shape(output)}")