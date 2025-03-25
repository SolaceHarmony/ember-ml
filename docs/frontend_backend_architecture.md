# Ember ML Frontend-Backend Architecture

## Overview

Ember ML implements a strict separation between frontend abstractions and backend implementations, allowing seamless switching between different computational backends (NumPy, PyTorch, MLX) while maintaining a consistent API.

## Frontend Architecture

### Core Abstractions

1. **Operations Module (`ops`)**
   - Provides backend-agnostic mathematical operations
   - All operations are accessed through the ops namespace
   - Example: `ops.sin()`, `ops.matmul()`, `ops.divide()`

2. **Tensor Module (`tensor`)**
   - Handles tensor creation and type conversion
   - Core method: `convert_to_tensor()`
   - Ensures consistent tensor handling across backends

### Frontend Patterns

#### 1. Operation Abstraction
```python
from ember_ml import ops
from ember_ml.nn import tensor

def process_data(x: tensor.Tensor) -> tensor.Tensor:
    """Apply sinusoidal transformation to input tensor.
    
    Args:
        x: Input tensor to process
        
    Returns:
        Transformed tensor
    """
    x = tensor.convert_to_tensor(x)
    return ops.sin(x)  # Backend-agnostic operation
```

#### 2. Tensor Conversion
```python
from ember_ml import ops
from ember_ml.nn import tensor

def normalize_data(x: tensor.Tensor) -> tensor.Tensor:
    """Normalize input tensor to [0, 1] range.
    
    Args:
        x: Input tensor with values in [0, 255]
        
    Returns:
        Normalized tensor with values in [0, 1]
    """
    x = tensor.convert_to_tensor(x)
    return ops.divide(x, ops.cast(255.0, x.dtype))
```

#### 3. Feature Extraction Pattern
```python
from ember_ml import ops
from ember_ml.nn import tensor
from typing import Optional

class FeatureExtractor:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
    
    def extract(self, input_data: tensor.Tensor, mask: Optional[tensor.Tensor] = None) -> tensor.Tensor:
        """Extract features from input data.
        
        Args:
            input_data: Raw input tensor
            mask: Optional mask tensor
            
        Returns:
            Processed feature tensor
        """
        x = tensor.convert_to_tensor(input_data)
        if mask is not None:
            mask = tensor.convert_to_tensor(mask)
            x = ops.multiply(x, mask)
        return x
```

### Specialized Operation Patterns

#### 1. Wave Processing Pattern
```python
from ember_ml import ops
from ember_ml.nn import tensor

def process_wave(x: tensor.Tensor) -> tensor.Tensor:
    """Transform wave data using backend-agnostic operations.
    
    Args:
        x: Input wave tensor
        
    Returns:
        Transformed wave tensor
    """
    x = tensor.convert_to_tensor(x)
    window = ops.hann_window(1024)
    frames = ops.frame(x, frame_length=1024, hop_length=512)
    return ops.stft(ops.multiply(frames, window))
```

#### 2. Scatter Operation Pattern
```python
def scatter_mean(values: tensor.Tensor, indices: tensor.Tensor, 
                dim_size: Optional[int] = None) -> tensor.Tensor:
    """Scatter-mean operation with backend agnosticism.
    
    Args:
        values: Source tensor containing values to scatter
        indices: Target indices for scattering
        dim_size: Optional size of output dimension
        
    Returns:
        Tensor with mean-aggregated scattered values
    """
    values = tensor.convert_to_tensor(values)
    indices = tensor.convert_to_tensor(indices)
    
    # First compute sum
    sum_result = ops.scatter(values, indices, dim_size, "add")
    
    # Then compute count using ones
    ones = ops.ones_like(values)
    count = ops.scatter(ones, indices, dim_size, "add")
    
    # Avoid division by zero
    count = ops.where(count == 0, ops.ones_like(count), count)
    
    return ops.divide(sum_result, count)
```

#### 3. Neural Specialization Pattern
```python
class SpecializedProcessor:
    """Backend-agnostic specialized neural processing."""
    
    def __init__(self, role: str = "default"):
        self.role = role
        
    def process(self, x: tensor.Tensor) -> tensor.Tensor:
        """Process input based on specialization role."""
        x = tensor.convert_to_tensor(x)
        
        if self.role == "memory":
            # Slower decay for memory
            return ops.multiply(x, 0.95)
        elif self.role == "inhibition":
            # Signal dampening
            return ops.multiply(x, -0.5)
        else:
            return x
```

### Advanced Operation Patterns

#### 1. Scatter Operations
```python
from ember_ml import ops
from ember_ml.nn import tensor

def scatter_example(data: tensor.Tensor, indices: tensor.Tensor) -> tensor.Tensor:
    """Demonstrate backend-agnostic scatter operation.
    
    Args:
        data: Source tensor to scatter
        indices: Target indices for scattering
        
    Returns:
        Scattered tensor
    """
    data = tensor.convert_to_tensor(data)
    indices = tensor.convert_to_tensor(indices)
    return ops.scatter(data, indices, aggr="add")
```

#### 2. Wave Processing
```python
from ember_ml import ops
from ember_ml.nn import tensor

def process_waveform(x: tensor.Tensor) -> tensor.Tensor:
    """Transform waveform using backend-agnostic operations.
    
    Args:
        x: Input waveform tensor
        
    Returns:
        Processed waveform tensor
    """
    x = tensor.convert_to_tensor(x)
    window = ops.hann_window(1024)
    return ops.stft(ops.multiply(x, window))
```

## Frontend-Backend Interaction

### 1. Operation Dispatch
- Frontend calls are routed to appropriate backend implementations
- Backend selection is handled through environment variables
- No direct backend imports in frontend code
- Operations are lazy-evaluated when possible

### 2. Type Safety
- Strong type annotations throughout
- Backend-agnostic tensor types
- No precision-reducing casts

### 3. Backend Independence
Frontend code remains pure by:
- Never importing backend libraries directly
- Using only the ops abstraction layer
- Maintaining backend-agnostic data types

## Best Practices

1. **Always Use Abstraction Layer**
   ```python
   # ✅ Correct - Backend agnostic
   from ember_ml import ops
   result = ops.matmul(a, b)

   # ✅ Correct - With type hints
   def compute(x: tensor.Tensor, y: tensor.Tensor) -> tensor.Tensor:
       return ops.matmul(x, y)

   # ❌ Wrong - Direct backend usage
   import numpy as np  # Never import backends directly
   result = np.matmul(a, b)
   ```

2. **Tensor Conversion**
   ```python
   # ✅ Correct
   x = tensor.convert_to_tensor(input_data)

   # ❌ Wrong
   x = np.array(input_data)
   ```

3. **Type Safety**
   ```python
   # ✅ Correct
   def process(x: tensor.Tensor) -> tensor.Tensor:
       return ops.relu(x)

   # ❌ Wrong
   def process(x):
       return x.numpy() * 2
   ```

## Implementation Guidelines

1. Keep all frontend code backend-agnostic
2. Use proper type annotations
3. Convert inputs using tensor.convert_to_tensor()
4. Use ops module for all operations
5. Never import backend libraries directly
6. Maintain comprehensive documentation

## Testing Considerations

1. Test with all supported backends
2. Verify type safety
3. Check edge cases
4. Ensure backend independence

## Testing Strategy

### 1. Backend Verification
```python
def test_backend_equivalence():
    """Verify operation equivalence across backends."""
    x = tensor.ones((10,))
    
    # Test with each backend
    for backend in ["numpy", "torch", "mlx"]:
        ops.set_backend(backend)
        result = ops.scatter(x, tensor.array([0, 1, 0]))
        # Results should match within numerical precision
```

### 2. Edge Case Testing
```python
def test_scatter_edge_cases():
    """Test scatter operations with edge cases."""
    # Empty tensor case
    empty = tensor.array([])
    indices = tensor.array([], dtype=tensor.int32)
    result = ops.scatter(empty, indices)
    
    # Large dimension case
    large = tensor.ones((10000,))
    sparse_indices = tensor.array([0, 9999])
    result = ops.scatter(large, sparse_indices)
```

### 3. Type Safety Verification
```python
def verify_type_safety():
    """Verify type safety across operations."""
    # Input type checking
    x = tensor.ones((10,), dtype=tensor.float32)
    y = tensor.ones((10,), dtype=tensor.float16)
    
    # Should raise TypeError for mismatched precision
    try:
        ops.matmul(x, y)
    except TypeError:
        pass
```

## Common Pitfalls to Avoid

1. Direct backend imports
2. Using backend-specific features
3. Mixing backend types
4. Precision-reducing casts
5. Direct Python operators on tensors

## Future Considerations

1. Adding new backends
2. Extending the ops interface
3. Performance optimizations
4. Type system improvements

## Framework-Wide Architecture

### 1. Neural Network Components

The neural network components follow strict frontend-backend separation:

#### Layer Primitives
```python
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.layers import base

class ChannelMixingLayer(base.Layer):
    """Base class for channel-mixing layers."""
    
    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        x = tensor.convert_to_tensor(x)
        # Layer logic using only ops abstraction
        return x

class SequenceMixingLayer(base.Layer):
    """Base class for sequence-mixing layers."""
    
    def forward(self, x: tensor.Tensor, mask: Optional[tensor.Tensor] = None) -> tensor.Tensor:
        x = tensor.convert_to_tensor(x)
        if mask is not None:
            mask = tensor.convert_to_tensor(mask)
            x = ops.multiply(x, mask)
        return x
```

### 2. Registry System

The registry system maintains backend independence:

```python
from ember_ml.registry import Registry, register

@register("models/transformer")
class TransformerConfig:
    """Configuration for Transformer model."""
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        
    def create(self, backend_context: Any) -> "TransformerModel":
        """Create model instance using current backend."""
        return TransformerModel(self.hidden_size)
```

### 3. Feature Extraction Framework

Feature extraction maintains backend agnosticism:

```python
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.features import base

class TimeSeriesFeatures(base.FeatureExtractor):
    """Extract features from time series data."""
    
    def extract(self, data: tensor.Tensor) -> tensor.Tensor:
        x = tensor.convert_to_tensor(data)
        # Use ops for all computations
        mean = ops.reduce_mean(x, axis=1, keepdims=True)
        std = ops.reduce_std(x, axis=1, keepdims=True)
        return ops.divide(ops.subtract(x, mean), std)
```

### 4. Model Architecture

Models follow the same frontend-backend separation:

```python
from ember_ml import ops
from ember_ml.nn import tensor, Model
from ember_ml.nn.layers import channel_mixing, sequence_mixing

class HybridModel(Model):
    """Model combining different layer types."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.channel_mixer = channel_mixing.MLP(config.hidden_size)
        self.sequence_mixer = sequence_mixing.Attention(config.hidden_size)
        
    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        x = tensor.convert_to_tensor(x)
        # Use only ops and registered layers
        x = self.channel_mixer(x)
        return self.sequence_mixer(x)
```

### 5. Distributed Training Support

Distributed operations maintain backend independence:

```python
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.distributed import strategy

class DataParallelStrategy(strategy.Strategy):
    """Data-parallel training strategy."""
    
    def distribute(self, batch: tensor.Tensor) -> List[tensor.Tensor]:
        batch = tensor.convert_to_tensor(batch)
        # Split using ops
        return ops.split(batch, self.num_devices)
```

### 6. Memory Management

Backend-agnostic memory management:

```python
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.memory import manager

class GradientCheckpointing:
    """Backend-agnostic gradient checkpointing."""
    
    def checkpoint(self, fn: Callable, *args) -> tensor.Tensor:
        # Convert all inputs
        args = [tensor.convert_to_tensor(arg) for arg in args]
        return manager.checkpoint_function(fn, *args)
```

## Implementation Strategy

### Component Hierarchy
1. Core Operations (`ops`)
2. Tensor Management (`tensor`)
3. Layer Primitives
4. Block Architecture
5. Model System
6. Training Framework
7. Distribution System

### Backend Integration Points
1. Tensor Creation/Conversion
2. Mathematical Operations
3. Device Management
4. Memory Optimization
5. Distribution Coordination

## Testing Matrix

| Component | Test Focus | Verification |
|-----------|------------|--------------|
| Ops | Backend Equivalence | Output matching |
| Tensor | Type Safety | Conversion correctness |
| Layers | Backend Independence | No leakage |
| Models | Cross-Backend | Training/Inference |
| Distributed | Coordination | Sync behavior |

## Migration Guidelines

1. Replace direct backend calls with `ops`
2. Convert inputs with `tensor.convert_to_tensor()`
3. Use backend-agnostic configuration
4. Implement backend-specific optimizations behind abstraction layer
5. Test across all supported backends

## Backend-Specific Optimizations

### 1. MLX Backend Optimizations
```python
def optimize_scatter_mlx(values, indices, dim_size):
    """MLX-specific scatter optimization."""
    # Convert at boundaries only
    values = tensor.convert_to_tensor(values)
    indices = tensor.convert_to_tensor(indices)
    
    # Use MLX's native scatter
    return mx.scatter(values, indices, dim_size)
```

### 2. PyTorch Backend Optimizations
```python
def optimize_scatter_torch(values, indices, dim_size):
    """PyTorch-specific scatter optimization."""
    values = tensor.convert_to_tensor(values)
    indices = tensor.convert_to_tensor(indices).long()
    
    # Use PyTorch's scatter_add_
    output = torch.zeros(dim_size)
    return output.scatter_add_(0, indices, values)
```

### 3. NumPy Backend Optimizations
```python
def optimize_scatter_numpy(values, indices, dim_size):
    """NumPy-specific scatter optimization."""
    values = tensor.convert_to_tensor(values)
    indices = tensor.convert_to_tensor(indices)
    
    # Use direct indexing for better performance
    output = np.zeros(dim_size)
    np.add.at(output, indices, values)
    return output
```
