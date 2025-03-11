from ember_ml import ops, set_backend
from ember_ml.nn.modules import AutoNCP
import importlib

def test_auto_ncp(backend_name):
    print(f"\n=== Testing with {backend_name} backend ===")
    
    # Set the backend
    set_backend(backend_name)
    
    # Import the backend's config module
    config = importlib.import_module(f'ember_ml.backend.{backend_name}.config')
    print(f"Device: {config.DEFAULT_DEVICE}")
    
    # Create an AutoNCP module
    auto_ncp = AutoNCP(
        units=15,
        output_size=5,
        sparsity_level=0.5,
        activation="tanh",
        use_bias=True
    )
    
    # Print the AutoNCP's parameters
    print("\nAutoNCP parameters:")
    for name, param in auto_ncp._parameters.items():
        print(f"- {name}: {param}")
    
    # Create input
    batch_size = 2
    inputs = ops.random_normal((batch_size, 15))
    
    # Forward pass
    output = auto_ncp(inputs)
    
    print(f"\nInput shape: {ops.shape(inputs)}")
    print(f"Output shape: {ops.shape(output)}")

# Test with each backend
backends = ['numpy', 'torch', 'mlx']
for backend in backends:
    try:
        test_auto_ncp(backend)
    except Exception as e:
        print(f"\nError testing {backend} backend: {str(e)}")