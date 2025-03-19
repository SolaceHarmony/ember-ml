# Ember ML Code Mode Rules

## Core Principles

As a Code mode for the Ember ML project, you must strictly adhere to the following principles:

1. **Backend Purity**: All code must be backend-agnostic, using the ops abstraction layer instead of direct framework calls
2. **Code Style**: Follow PEP 8 and project-specific style guidelines
3. **Type Safety**: Use proper type annotations for all functions and methods
4. **Documentation**: Provide comprehensive docstrings for all code
5. **Testing**: Ensure all code is testable and has corresponding tests

## CRITICAL: EmberLint Validation

**EVERY CODE CHANGE MUST BE VALIDATED WITH EMBERLINT BEFORE SUBMISSION**

The `utils/emberlint.py` tool is the definitive authority on code correctness in the Ember ML project. You must run this tool on every file you modify and ensure it passes all checks before considering your work complete.

```bash
python utils/emberlint.py path/to/modified/file.py --verbose
```

## Backend Purity Requirements

### 1. NO DIRECT NUMPY USAGE

❌ **FORBIDDEN**:
- Importing NumPy directly (`import numpy` or `from numpy import ...`)
- Using NumPy functions or methods (`np.array()`, `np.sin()`, etc.)
- Converting tensors to NumPy arrays (`.numpy()`, `np.array(tensor)`)

✅ **REQUIRED**:
- Import ops from ember_ml (`from ember_ml import ops`)
- Use ops functions for all mathematical operations (`ops.sin()`, `ops.matmul()`)
- Use ops for tensor creation and manipulation (`tensor.convert_to_tensor()`)

**EXCEPTION**: NumPy usage is permitted ONLY for visualization/plotting libraries that specifically require it, and ONLY after thorough testing to confirm that it is required (causes an exception or other issue when using the abstraction layer). Even in these cases, the code should be isolated and clearly documented.

### 2. NO PRECISION-REDUCING CASTS

❌ **FORBIDDEN**:
- Using `float()` or `int()` casts that may reduce precision
- Hardcoding data types that may not be compatible with all backends

✅ **REQUIRED**:
- Use `ops.cast()` with appropriate dtype
- Use dtype constants from `ember_ml.ops.dtypes`

### 3. NO DIRECT PYTHON OPERATORS

❌ **FORBIDDEN**:
- Using Python operators directly on tensors (`+`, `-`, `*`, `/`, etc.)
- Using Python comparison operators on tensors (`<`, `>`, `==`, etc.)

✅ **REQUIRED**:
- Use ops functions for all operations (`ops.add()`, `ops.subtract()`, etc.)
- Use ops functions for comparisons (`ops.equal()`, `ops.greater()`, etc.)

### 4. NO DIRECT BACKEND ACCESS

❌ **FORBIDDEN**:
- Bypassing frontend abstractions to use backends directly
- Implementing backend-specific code outside the backend directory
- Using backend-specific features in frontend code

✅ **REQUIRED**:
- Always use the ops and nn abstraction layers rather than backend implementation of torch, mlx, numpy, etc.
- Keep all backend implementations within the backend directory
- Use backend-agnostic code in all frontend components

## Backend Abstraction Architecture

### Frontend-Backend Separation

The Ember ML framework uses a strict separation between frontend abstractions and backend implementations:

1. **Frontend Abstractions**: The `ops` and `nn.*` (and other folders and classes) are generally abstract interfaces (stubs) that define the API but do not contain actual implementations
2. **Backend Implementations**: The actual implementations reside in the backend directory, with specific implementations for each supported backend (NumPy, PyTorch, MLX)
3. **Dispatch Mechanism**: The frontend abstractions dispatch calls to the appropriate backend implementation based on the currently selected backend

This architecture allows MLX, PyTorch, and NumPy layers, operations, solvers, etc. to be used natively through a consistent API.

### Backend Folder Structure

The backend implementations are organized in a modular folder structure:

```
ember_ml/backend/
├── numpy/             # NumPy backend implementations
│   ├── tensor_ops.py  # Tensor operations
│   ├── math_ops.py    # Math operations
│   ├── random_ops.py  # Random operations
│   ├── io_ops.py      # I/O operations
│   └── ...
├── torch/             # PyTorch backend implementations
│   ├── tensor_ops.py
│   ├── math_ops.py
│   ├── random_ops.py
│   ├── io_ops.py
│   └── ...
├── mlx/               # MLX backend implementations
│   ├── tensor_ops.py
│   ├── math_ops.py
│   ├── random_ops.py
│   ├── io_ops.py
│   └── ...
└── ...
```

Each backend folder contains implementation files that correspond to the operation categories defined in the frontend interfaces. This modular structure makes it easier to maintain and extend the backend implementations.

### Interface Definition and Implementation

For each operation category, there is:

1. **Interface Definition**: An abstract class in `ember_ml/ops/interfaces/` that defines the API
2. **Frontend Exposure**: Functions in `ember_ml/ops/__init__.py` that expose the operations to users
3. **Backend Implementation**: Concrete implementations in each backend folder

For example, for I/O operations:
- Interface: `ember_ml/ops/interfaces/io_ops.py` defines the `IOOps` interface
- Frontend: `ember_ml/ops/__init__.py` exposes `save()` and `load()` functions
- Backends: `ember_ml/backend/numpy/io_ops.py`, `ember_ml/backend/torch/io_ops.py`, and `ember_ml/backend/mlx/io_ops.py` provide the implementations

### Key Architectural Rules

1. **Frontend-Only Rule**: Frontend code must NEVER contain backend-specific implementations
2. **Backend-Only Rule**: Backend implementations must ONLY reside in the backend directory
3. **Abstraction-Only Rule**: All interaction with tensors and neural network components must go through the abstraction layer
4. **No Mixing Rule**: Never mix different backends in the same computation graph

### Example of Correct Architecture

```python
# Frontend abstraction (in ops/math.py)
def sin(x):
    """Compute sine of x element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sine of x
    """
    x = convert_to_tensor(x)
    return get_backend().sin(x)

# Backend implementation (in backend/numpy_backend.py)
def sin(x):
    """NumPy implementation of sine."""
    return np.sin(x)

# Backend implementation (in backend/torch_backend.py)
def sin(x):
    """PyTorch implementation of sine."""
    return torch.sin(x)

# Backend implementation (in backend/mlx_backend.py)
def sin(x):
    """MLX implementation of sine."""
    return mlx.core.sin(x)
```

## Code Structure and Organization

### Module Organization

Follow the established module structure:

```
ember_ml/
├── backend/       # Backend abstraction system
├── ops/           # Operations (math, tensor, random)
├── nn/            # Neural network components
├── features/      # Feature extraction
├── models/        # Model implementations
└── ...            # Other modules
```

### File Organization

Each file should be organized as follows:

1. Module docstring
2. Imports (grouped by standard library, third-party, and internal)
3. Constants and global variables
4. Classes
5. Functions
6. Main execution block (if applicable)

## Documentation Standards

### Docstrings

Use Google-style docstrings for all functions, classes, and modules:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Short description of the function.
    
    Longer description explaining the function's purpose and behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
    """
```

### Type Annotations

Use type annotations for all function parameters and return values:

```python
def process_data(data: Union[List[float], np.ndarray, torch.Tensor]) -> ops.Tensor:
    """Process the input data."""
    # Implementation
```

## Testing Requirements

### Unit Tests

Every component must have corresponding unit tests that:

1. Test the component in isolation
2. Cover all code paths and edge cases
3. Test with different backends (NumPy, PyTorch, MLX)

### Integration Tests

Components that interact with other parts of the system must have integration tests that:

1. Test the component in the context of the larger system
2. Verify correct behavior across component boundaries

## Common Pitfalls to Avoid

### 1. Backend Leakage

❌ **WRONG**:
```python
import numpy as np

def process_data(data):
    return np.sin(data)
```

✅ **CORRECT**:
```python
from ember_ml import ops

def process_data(data):
    tensor = tensor.convert_to_tensor(data)
    return ops.sin(tensor)
```

### 2. Precision Loss

❌ **WRONG**:
```python
def normalize(x):
    return float(x) / 255.0
```

✅ **CORRECT**:
```python
from ember_ml import ops

def normalize(x):
    x_tensor = tensor.convert_to_tensor(x)
    return ops.divide(x_tensor, tensor.convert_to_tensor(255.0))
```

### 3. Direct Operators

❌ **WRONG**:
```python
def add_tensors(a, b):
    return a + b
```

✅ **CORRECT**:
```python
from ember_ml import ops

def add_tensors(a, b):
    return ops.add(a, b)
```

### 4. Backend Implementation Outside Backend Directory

❌ **WRONG**:
```python
# In features/extractor.py
import torch

def extract_features(data):
    # Direct PyTorch implementation outside backend directory
    return torch.nn.functional.relu(data)
```

✅ **CORRECT**:
```python
# In features/extractor.py
from ember_ml import ops

def extract_features(data):
    # Use ops abstraction
    return ops.relu(data)
```

## Implementation Guidelines

### I/O Operations

When implementing I/O operations:

1. **Use the Abstraction Layer**: Always use the `ops.save()` and `ops.load()` functions from the ops abstraction layer
2. **Never Import Backend Directly**: Never use `from ember_ml import backend as K` or similar direct backend imports
3. **Implement in Backend Folders**: Implement I/O operations in the appropriate backend folder (e.g., `backend/numpy/io_ops.py`)
4. **Define Interfaces**: Define interfaces in `ops/interfaces/io_ops.py` before implementing in backends
5. **Expose in ops/__init__.py**: Expose I/O functions in `ops/__init__.py` using the dispatch mechanism

### Feature Extraction

When implementing feature extraction components:

1. Use the `ColumnFeatureExtractor` pattern for tabular data
2. Use the `TemporalStrideProcessor` for time series data
3. Ensure all components are backend-agnostic

### Neural Networks

When implementing neural network components:

1. Use the `nn` module for layers and cells
2. Use the `wirings` module for connectivity patterns
3. Ensure all components work with the backend abstraction system

### RBM and Liquid Networks

When working with RBMs and liquid networks:

1. Use the established patterns in `models/rbm.py` and `models/liquid.py`
2. Ensure compatibility with the feature extraction pipeline
3. Follow the CfC architecture for temporal processing

## Development Process Guidelines

### Task Management

1. **Use Checklists**: Always create and use checklists of tasks or files that need to be modified
   - Create a checklist at the beginning of each task
   - Check off items as they are completed
   - Review the checklist before marking the task as complete

2. **Completion Verification**: Never mark anything as done unless it's truly complete
   - All requirements must be fully implemented
   - All tests must pass
   - All documentation must be updated
   - If you are unsure about completion status, switch to Ask mode for clarification

3. **Code Inspection**: Always verify implementation details when in doubt
   - Check backend and other implementation signatures at site-packages
   - Inspect the actual code if you have the slightest doubt about behavior
   - Verify that your implementation matches the expected interface and behavior

### Example Checklist Format

```
## Task: Implement new feature X

- [ ] Review existing code and documentation
- [ ] Create implementation plan
- [ ] Implement core functionality
- [ ] Add type annotations
- [ ] Write comprehensive docstrings
- [ ] Create unit tests
- [ ] Create integration tests
- [ ] Run emberlint on all modified files
- [ ] Verify backend compatibility
- [ ] Update documentation
```

## Final Checklist

Before submitting any code, verify that:

1. ✅ EmberLint passes with no errors - run `python utils/emberlint.py path/to/file.py --verbose`
2. ✅ All functions have proper type annotations - use `mypy` for static type checking
3. ✅ All functions have comprehensive docstrings - follow the Google-style format
4. ✅ No direct NumPy usage or other backend-specific code - use the ops abstraction layer
5. ✅ No precision-reducing casts - never use `float()` or `int()`
6. ✅ No direct Python operators on tensors - use ops functions for all operations
7. ✅ Code follows the established module and file organization - frontend abstractions, backend implementations, etc.
8. ✅ No backend implementations outside the backend directory - all backend code should be in backend/
9. ✅ All tensor operations go through the ops abstraction layer - no direct backend access
10. ✅ Unit tests cover all code paths - test with different backends
11. ✅ Integration tests verify component interactions  - test across component boundaries
12. ✅ All tasks in your checklist are completed and verified

**REMEMBER: ALWAYS RUN EMBERLINT ON EVERY FILE YOU MODIFY**

The quality and consistency of the Ember ML codebase depends on strict adherence to these rules. Backend purity is especially critical to ensure the framework works consistently across different computational backends.

## Step-by-Step Guide for Adding a New Function

When adding a new function to the Ember ML library, follow these steps precisely to ensure proper implementation across the frontend abstraction and backend implementations:

### 1. Define the Interface

1. Identify the appropriate interface file in `ember_ml/ops/interfaces/` (e.g., `math_ops.py` for mathematical operations)
2. Add the new method to the abstract class with:
   - Proper type annotations
   - Comprehensive docstring following Google style
   - `@abstractmethod` decorator
   - `pass` statement for the implementation

Example:
```python
@abstractmethod
def new_function(self, x: Any, y: Any) -> Any:
    """
    Short description of the function.
    
    Longer description explaining the purpose and behavior.
    
    Args:
        x: Description of x
        y: Description of y
        
    Returns:
        Description of return value
    """
    pass
```

### 2. Expose the Function in the Frontend

1. Open `ember_ml/ops/__init__.py`
2. Add a lambda function that calls the appropriate backend method:
   ```python
   new_function = lambda *args, **kwargs: math_ops().new_function(*args, **kwargs)
   ```
3. Add the function name to the `__all__` list to expose it in the public API

### 3. Implement the Function in Each Backend

For each backend (NumPy, PyTorch, MLX), implement the function in the appropriate file:

#### NumPy Backend (`ember_ml/backend/numpy/math_ops.py`):
1. Add the function with proper type annotations and docstring
2. Use `convert_to_tensor` to ensure inputs are NumPy arrays
3. Implement using NumPy functions
4. Add the method to the corresponding Ops class (e.g., `NumpyMathOps`)

#### PyTorch Backend (`ember_ml/backend/torch/math_ops.py`):
1. Add the function with proper type annotations and docstring
2. Use `convert_to_tensor` to ensure inputs are PyTorch tensors
3. Implement using PyTorch functions
4. Add the method to the corresponding Ops class (e.g., `TorchMathOps`)

#### MLX Backend (`ember_ml/backend/mlx/math_ops.py`):
1. Add the function with proper type annotations and docstring
2. Use `convert_to_tensor` to ensure inputs are MLX arrays
3. Implement using MLX functions
4. Add the method to the corresponding Ops class (e.g., `MLXMathOps`)

### 4. Validate with EmberLint

1. Run EmberLint on all modified files:
   ```bash
   python utils/emberlint.py ember_ml/ops/interfaces/math_ops.py --verbose
   python utils/emberlint.py ember_ml/ops/__init__.py --verbose
   python utils/emberlint.py ember_ml/backend/numpy/math_ops.py --verbose
   python utils/emberlint.py ember_ml/backend/torch/math_ops.py --verbose
   python utils/emberlint.py ember_ml/backend/mlx/math_ops.py --verbose
   ```
2. Fix any issues reported by EmberLint

### 5. Test the Function

1. Create a simple test that uses the ops abstraction layer (NOT direct backend calls):
   ```python
   from ember_ml import ops
   result = ops.new_function(tensor.convert_to_tensor([1, 2, 3]), tensor.convert_to_tensor([4, 5, 6]))
   print(result)
   ```
2. Verify the function works as expected

### 6. Add Unit Tests

1. Add unit tests in the appropriate test file
2. Test with all backends
3. Test edge cases and error conditions

### 7. Update Documentation

1. Add the function to the API documentation
2. Provide examples of usage

By following these steps consistently, you ensure that new functions maintain the backend purity and architectural integrity of the Ember ML framework.