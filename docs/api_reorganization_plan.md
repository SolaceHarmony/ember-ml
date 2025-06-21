# Ember ML API Reorganization Plan

## Overview

This document outlines the plan to reorganize the Ember ML API to create a more intuitive and consistent user experience. The primary goals are:

1. Move tensor creation functions to the top level for easier access
2. Consolidate operations in a logical, categorized structure
3. Simplify the backend system while maintaining flexibility
4. Create a more intuitive experience that aligns with user expectations

## Current Structure Issues

The current API structure has led to confusion and usability issues:

1. Operations are split between `ember_ml.nn.tensor` and `ember_ml.ops`
2. Tensor creation and manipulation are separated
3. Users (including maintainers) struggle to remember where to find specific operations
4. The structure doesn't align with common expectations based on other libraries like PyTorch, NumPy, and MLX

```mermaid
graph TD
    A[ember_ml] --> B[ops]
    A --> C[nn]
    C --> D[tensor]
    
    B --> B1[ops.add, ops.matmul, etc.]
    B --> B2[ops.stats]
    B --> B3[ops.linearalg]
    B --> B4[ops.bitwise]
    
    D --> D1[tensor.array, tensor.ones, etc.]
    D --> D2[tensor.reshape, tensor.concatenate, etc.]
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style B1 fill:#f9f,stroke:#333,stroke-width:1px
    style D1 fill:#bbf,stroke:#333,stroke-width:1px
    style D2 fill:#bbf,stroke:#333,stroke-width:1px
```

## Target Structure

The proposed structure consolidates operations and brings tensor creation to the top level:

```mermaid
graph TD
    A[ember_ml] --> A1[array, ones, zeros, etc.]
    A --> A2[add, subtract, multiply, etc.]
    A --> B[linalg]
    A --> C[stats]
    A --> D[random]
    A --> E[activations]
    A --> F[nn]
    
    B --> B1[svd, qr, etc.]
    C --> C1[mean, std, etc.]
    D --> D1[normal, uniform, etc.]
    E --> E1[relu, sigmoid, etc.]
    F --> F1[modules]
    
    style A1 fill:#bfb,stroke:#333,stroke-width:1px
    style A2 fill:#bfb,stroke:#333,stroke-width:1px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
```

## Key API Changes

### Current API Usage

```python
from ember_ml import tensor
from ember_ml import ops

# Create tensors
x = tensor.array([1, 2, 3])
zeros = tensor.zeros((2, 3))

# Perform operations
y = ops.add(x, x)
z = tensor.reshape(y, (3, 1))  # Some ops are in tensor module
```

### New API Usage

```python
import ember_ml as em

# Create tensors at top level
x = em.array([1, 2, 3])
zeros = em.zeros((2, 3))

# Perform operations at top level
y = em.add(x, x)
z = em.reshape(y, (3, 1))  # All ops at top level

# Specialized operations in organized submodules
result = em.linalg.svd(z)
```

## Implementation Plan

The implementation will proceed in phases to minimize disruption while ensuring a smooth transition.

### Phase 1: New Structure Creation

```mermaid
gantt
    title Phase 1: New Structure Creation
    dateFormat  YYYY-MM-DD
    section Tasks
    Create new directory structure      :a1, 2025-06-24, 1d
    Update ember_ml/__init__.py         :a2, after a1, 1d
    Create submodule __init__ files     :a3, after a2, 1d
    Create basic directory skeleton     :a4, after a3, 1d
```

**Tasks:**
1. Create new directory structure for top-level modules:
   - `ember_ml/linalg/`
   - `ember_ml/stats/`
   - `ember_ml/random/`
   - `ember_ml/activations/`

2. Create skeleton `__init__.py` files with import placeholders.

3. Update `ember_ml/__init__.py` to include imports for new organization.

### Phase 2: Move Tensor Operations

```mermaid
gantt
    title Phase 2: Move Tensor Operations
    dateFormat  YYYY-MM-DD
    section Tasks
    Move tensor creation functions      :b1, 2025-06-25, 2d
    Update imports in modules           :b2, after b1, 1d
    Move operations to proper categories:b3, after b2, 2d
    Update proxy module mappings        :b4, after b3, 1d
```

**Tasks:**
1. Move tensor creation functions from `nn.tensor.common` to top level.
2. Update imports in all modules to use new API.
3. Reorganize operations based on categories.
4. Update proxy module mappings to reflect new organization.

### Phase 3: Backend Implementation Updates

```mermaid
gantt
    title Phase 3: Backend Implementation Updates
    dateFormat  YYYY-MM-DD
    section Tasks
    Flatten backend structure           :c1, 2025-06-30, 2d
    Update import paths                 :c2, after c1, 1d
    Update backend registry system      :c3, after c2, 2d
    Create categorized registration     :c4, after c3, 1d
```

**Tasks:**
1. Flatten backend structure by merging `backend/*/tensor` into main backend folders.
2. Update import paths throughout the codebase.
3. Enhance registry to support categories.
4. Create registration functions for categorized operations.

### Phase 4: Activation Functions Integration

```mermaid
gantt
    title Phase 4: Activation Functions Integration
    dateFormat  YYYY-MM-DD
    section Tasks
    Create activations module           :d1, 2025-07-05, 1d
    Move functional activations         :d2, after d1, 1d
    Update activation module class      :d3, after d2, 1d
    Update dynamic aliasing system      :d4, after d3, 1d
```

**Tasks:**
1. Create dedicated top-level `activations` module.
2. Move functional activation operations from backend.
3. Update module classes in `nn.modules.activations`.
4. Update dynamic aliasing system for activation functions.

### Phase 5: Testing and Documentation

```mermaid
gantt
    title Phase 5: Testing and Documentation
    dateFormat  YYYY-MM-DD
    section Tasks
    Update tests                        :e1, 2025-07-08, 2d
    Expand test coverage                :e2, after e1, 2d
    Update API documentation            :e3, after e2, 2d
    Create migration guide              :e4, after e3, 1d
    Update examples                     :e5, after e4, 1d
```

**Tasks:**
1. Update import paths in all tests.
2. Add new tests for the reorganized API.
3. Update API documentation to reflect new structure.
4. Create migration guide for users.
5. Update examples to use new API pattern.

### Phase 6: Backward Compatibility (Optional)

```mermaid
gantt
    title Phase 6: Backward Compatibility
    dateFormat  YYYY-MM-DD
    section Tasks
    Add deprecation warnings            :f1, 2025-07-15, 1d
    Create compatibility imports        :f2, after f1, 2d
    Document migration timeline         :f3, after f2, 1d
```

**Tasks:**
1. Add deprecation warnings to old API paths.
2. Create temporary compatibility imports.
3. Document migration timeline for API changes.

## File Modification List

### Core Structure Files

1. **`ember_ml/__init__.py`**
   - Add imports for all top-level functions
   - Re-export submodule functions
   - Create new API patterns

2. **`ember_ml/ops/__init__.py`**
   - Update to export top-level operations
   - Reorganize operations by category

3. **`ember_ml/ops/proxy.py`**
   - Update proxy module mappings
   - Enhance for categorized operation support

### Backend System Files

1. **`ember_ml/backend/__init__.py`**
   - Update backend initialization
   - Modify backend registration

2. **`ember_ml/backend/registry.py`**
   - Enhance registry to support categories
   - Update registration mechanisms

3. **Backend-specific implementation files**
   - Flatten directory structure
   - Move tensor operations to appropriate categories

### Tensor Files

1. **`ember_ml/nn/tensor/__init__.py`**
   - Update to use new API
   - Add compatibility imports

2. **`ember_ml/nn/tensor/common/__init__.py`**
   - Move operations to appropriate new locations
   - Add compatibility imports

### New Module Files

1. **`ember_ml/linalg/__init__.py`**
   - Import linear algebra operations
   - Register with backend system

2. **`ember_ml/stats/__init__.py`**
   - Import statistics operations
   - Register with backend system

3. **`ember_ml/random/__init__.py`**
   - Import random operations
   - Register with backend system

4. **`ember_ml/activations/__init__.py`**
   - Import activation functions
   - Register with backend system

## Backend System Design

The updated backend system will support both direct access and categorized operations:

```mermaid
graph TD
    A[ember_ml API] --> B[Registry System]
    B --> C[Backend Selection]
    C --> D[NumPy Backend]
    C --> E[PyTorch Backend]
    C --> F[MLX Backend]
    
    B --> G[Direct Function Access]
    B --> H[Categorized Function Access]
    
    G --> I[em.array]
    G --> J[em.add]
    
    H --> K[em.linalg.svd]
    H --> L[em.stats.mean]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bfb,stroke:#333,stroke-width:2px
    style H fill:#bfb,stroke:#333,stroke-width:2px
```

## Example Migration

### Before:

```python
from ember_ml import tensor
from ember_ml import ops

x = tensor.ones((3, 3))
y = ops.matmul(x, x)
z = tensor.reshape(y, (9,))
```

### After:

```python
import ember_ml as em

x = em.ones((3, 3))
y = em.matmul(x, x)
z = em.reshape(y, (9,))

# Or with specialized imports:
from ember_ml import linalg
w = linalg.svd(y)
```

## Conclusion

This API reorganization will significantly improve the user experience by providing a more intuitive and consistent API that aligns with user expectations. By centralizing tensor operations at the top level while maintaining logically organized categories, we achieve both simplicity for common tasks and clarity for specialized operations.

The phased implementation approach allows for careful testing and documentation at each stage, minimizing the risk of disruption while moving toward a more maintainable and user-friendly API structure.
