# Backend Migration Plan

## Overview

According to the `.clinerules-code` file, backend-specific implementations should only reside in the backend directory, not in the ops directory. The current structure violates the Frontend-Only Rule and Backend-Only Rule.

This document outlines a plan to migrate all backend-specific implementations from the ops/torch, ops/mlx, and ops/numpy directories to the backend directory.

## Current Issues

The emberlint tool has identified the following issues:

1. **Frontend-Backend Separation Violations**: All files in ops/torch, ops/mlx, and ops/numpy directories contain backend-specific implementations, which violates the architectural rules.

2. **Backend Inconsistencies**: Many operations are implemented inconsistently across backends, with some operations missing in certain backends.

## Migration Strategy

We will use the `analyze_backend_operations.py` tool to help with the migration process. The tool can:

1. Identify operations that need to be migrated
2. Generate a migration plan
3. Migrate operations from ops to backend

### Step 1: Analyze Current State

Run the analyze_backend_operations.py tool to get a comprehensive view of the current state:

```bash
python utils/analyze_backend_operations.py --verbose
```

This will generate:
- A matrix of operations across backends
- A list of missing operations in each backend
- A list of operations to migrate

### Step 2: Generate Migration Plan

Generate a migration plan for each backend:

```bash
python utils/analyze_backend_operations.py --migrate --backend torch --dry-run --verbose
python utils/analyze_backend_operations.py --migrate --backend mlx --dry-run --verbose
python utils/analyze_backend_operations.py --migrate --backend numpy --dry-run --verbose
```

This will show which operations need to be migrated and how they will be migrated.

### Step 3: Migrate Operations

Migrate operations for each backend:

```bash
python utils/analyze_backend_operations.py --migrate --backend torch
python utils/analyze_backend_operations.py --migrate --backend mlx
python utils/analyze_backend_operations.py --migrate --backend numpy
```

### Step 4: Update Frontend Abstractions

After migrating all operations, update the frontend abstractions to dispatch to the backend implementations:

1. Update ops/tensor_ops.py
2. Update ops/math_ops.py
3. Update ops/random_ops.py
4. Update other ops modules

### Step 5: Test

Run tests to ensure functionality is preserved:

```bash
python -m pytest tests/
```

### Step 6: Remove Backend-Specific Directories

Once all operations have been migrated and tests pass, remove the backend-specific directories:

```bash
rm -rf emberharmony/ops/torch
rm -rf emberharmony/ops/mlx
rm -rf emberharmony/ops/numpy
```

### Step 7: Verify with EmberLint

Run emberlint to verify that there are no more frontend-backend separation violations:

```bash
python utils/emberlint.py emberharmony --frontend-backend-only --verbose
```

## Migration Details

### Torch Operations

Based on the analysis, there are 120 operations to migrate from ops/torch to backend/torch_backend.py. Some operations already exist in the backend file.

Example operations to migrate:
- squeeze
- full
- arange
- tile
- sin
- min
- get_available_devices
- random_poisson

### MLX Operations

Similar analysis needed for MLX operations.

### NumPy Operations

Similar analysis needed for NumPy operations.

## Frontend Abstraction Updates

After migrating operations, the frontend abstractions need to be updated to dispatch to the backend implementations. Here's an example of how a frontend abstraction should look:

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
```

## Timeline

1. **Week 1**: Analyze current state and generate migration plan
2. **Week 2**: Migrate Torch operations
3. **Week 3**: Migrate MLX operations
4. **Week 4**: Migrate NumPy operations
5. **Week 5**: Update frontend abstractions and test
6. **Week 6**: Remove backend-specific directories and verify with EmberLint

## Conclusion

This migration will ensure that the EmberHarmony codebase follows the architectural rules defined in the `.clinerules-code` file, with backend-specific implementations residing only in the backend directory and frontend code being backend-agnostic.