# Backend Purity Migration Checklist

## Top Priority: Migrate Backend-Specific Implementations to Backend Directory

According to the `.clinerules-code`, backend-specific implementations should only reside in the backend directory, not in the ops directory. The current structure violates the Frontend-Only Rule and Backend-Only Rule.

### Migration Tasks

- [ ] **Analyze Current Structure**
  - [ ] Identify all operations in ops/torch
  - [ ] Identify all operations in ops/mlx
  - [ ] Identify all operations in ops/numpy
  - [ ] Map these operations to their corresponding frontend abstractions

- [x] **Prepare Backend Directory**
  - [x] Ensure backend/torch_backend.py is ready for new implementations
  - [x] Ensure backend/mlx_backend.py is ready for new implementations
  - [x] Ensure backend/numpy_backend.py is ready for new implementations

- [x] **Migrate Torch Operations**
  - [x] Move tensor operations from ops/torch to backend/torch_backend.py
  - [x] Move math operations from ops/torch to backend/torch_backend.py
  - [x] Move random operations from ops/torch to backend/torch_backend.py
  - [x] Move other operations from ops/torch to backend/torch_backend.py
  - [x] Fix circular import issue by removing `from emberharmony import ops` from torch_backend.py
  - [x] Fix Python operator issues in the `split` function

- [x] **Migrate MLX Operations**
  - [x] Move tensor operations from ops/mlx to backend/mlx folder
  - [x] Move math operations from ops/mlx to backend/mlx folder
  - [x] Move random operations from ops/mlx to backend/mlx folder
  - [x] Move other operations from ops/mlx to backend/mlx folder
  - [x] Create MLXMathOps class in backend/mlx/math_ops.py (needed for ops/__init__.py)

- [ ] **Migrate NumPy Operations**
  - [ ] Move tensor operations from ops/numpy to backend/numpy folder
  - [ ] Move math operations from ops/numpy to backend/numpy folder
  - [ ] Move random operations from ops/numpy to backend/numpy folder
  - [ ] Move other operations from ops/numpy to backend/numpy folder

- [ ] **Update Frontend Abstractions**
  - [ ] Ensure ops/tensor_ops.py properly dispatches to backend implementations
  - [ ] Ensure ops/math_ops.py properly dispatches to backend implementations
  - [ ] Ensure ops/random_ops.py properly dispatches to backend implementations
  - [ ] Ensure other ops modules properly dispatch to backend implementations

- [ ] **Testing**
  - [x] Run unit tests for torch_backend.py (Fixed circular import issue)
  - [x] Run unit tests for mlx_backend.py (Fixed missing MLXMathOps class)
  - [x] Run unit tests for numpy_backend.py
  - [ ] Run integration tests to verify system behavior is preserved
  - [x] Test with different backends to ensure consistent behavior
  - [x] Implement missing Ops classes for MLX backend
  - [x] Implement missing Ops classes for PyTorch backend
  - [ ] Implement missing Ops classes for NumPy backend
  - [x] Refactor PyTorch backend into a modular folder structure
  - [x] Create folder structure for NumPy backend
  - [x] Create folder structure for MLX backend
  - [ ] Thoroughly test all tensor operations through ops interface
  - [ ] Thoroughly test all math operations through ops interface
  - [ ] Thoroughly test all random operations through ops interface
  - [ ] Thoroughly test all comparison operations through ops interface
  - [ ] Thoroughly test all device operations through ops interface
  - [ ] Thoroughly test all dtype operations through ops interface
  - [ ] Thoroughly test all solver operations through ops interface
  - [ ] Verify that no direct backend imports are used in tests

- [ ] **Cleanup**
  - [ ] Remove ops/torch directory after successful migration
  - [ ] Remove ops/mlx directory after successful migration
  - [ ] Remove ops/numpy directory after successful migration
  - [ ] Update documentation to reflect the new structure

- [x] **Validation**
  - [x] Run emberlint on all modified files
  - [x] Verify backend purity requirements are met
  - [x] Ensure no direct backend access in frontend code

## Remaining Issues

### Backend Persistence
- [x] Fix backend persistence issue where backend is reset to 'mlx' after reloading the module

### PyTorch Backend
- [x] Fix circular import issue by removing `from emberharmony import ops` from torch_backend.py
- [x] Fix Python operator issues in the `split` function
- [x] Implement missing TorchTensorOps class
- [x] Implement missing TorchDeviceOps, TorchRandomOps, TorchComparisonOps, TorchDTypeOps, and TorchSolverOps classes
- [x] Refactor PyTorch backend into a modular folder structure
- [ ] Address type annotation issues
- [ ] Fix style issues

### MLX Backend
- [x] Fix tensor conversion issues in the `to_numpy` function
- [ ] Address type annotation issues
- [ ] Fix style issues
- [x] Implement missing MLXMathOps class
- [x] Implement missing MLXTensorOps class
- [x] Implement missing MLXDeviceOps, MLXRandomOps, MLXComparisonOps, MLXDTypeOps, and MLXSolverOps classes
- [x] Create folder structure for MLX backend

### NumPy Backend
- [ ] Migrate all operations
- [ ] Fix tensor conversion issues
- [ ] Address type annotation issues
- [ ] Fix style issues
- [ ] Implement missing NumpyTensorOps class
- [ ] Implement missing NumpyDeviceOps, NumpyRandomOps, NumpyComparisonOps, NumpyDTypeOps, and NumpySolverOps classes
- [x] Create folder structure for NumPy backend

## Implementation Approach

1. **One Operation at a Time**: Migrate one operation at a time to ensure proper testing and validation
2. **Test After Each Migration**: Run tests after each operation is migrated to catch issues early
3. **Maintain Functionality**: Ensure the behavior of each operation remains the same after migration
4. **Follow Architectural Rules**:
   - Frontend code must NEVER contain backend-specific implementations
   - Backend implementations must ONLY reside in the backend directory
   - All interaction with tensors must go through the abstraction layer

## Example Migration Pattern

For each operation (e.g., `sin`):

1. Identify the implementation in ops/torch/math_ops.py, ops/mlx/math_ops.py, and ops/numpy/math_ops.py
2. Move the implementation to backend/torch_backend.py, backend/mlx_backend.py, and backend/numpy_backend.py
3. Update the frontend abstraction in ops/math_ops.py to dispatch to the backend implementation
4. Test to ensure functionality is preserved
5. Remove the original implementation from the ops directory

## Correct Architecture Example

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