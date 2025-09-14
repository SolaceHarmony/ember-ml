# Ember ML API Structure Visualization

## Current API Structure

The current Ember ML API structure divides operations between different modules, creating confusion about where to find specific functionality.

```mermaid
flowchart TD
    root[ember_ml] --> ops[ops]
    root --> nn[nn]
    nn --> tensor[tensor]
    nn --> modules[modules]
    
    ops --> math["ops.add, ops.matmul, etc."]
    ops --> stats["ops.stats"]
    ops --> linalg["ops.linearalg"]
    ops --> bitwise["ops.bitwise"]
    
    tensor --> creation["tensor.array, tensor.ones, etc."]
    tensor --> reshape["tensor.reshape, tensor.stack, etc."]
    
    modules --> act[modules.activations]
    act --> act_f["relu(), sigmoid(), etc."]
    act --> act_c["ReLU, Sigmoid classes"]
    
    classDef ops fill:#f96,stroke:#333,stroke-width:2px;
    classDef tensor fill:#69f,stroke:#333,stroke-width:2px;
    classDef modules fill:#9f6,stroke:#333,stroke-width:2px;
    
    class ops,math,stats,linalg,bitwise ops;
    class tensor,creation,reshape tensor;
    class modules,act,act_f,act_c modules;
```

## New API Structure

The proposed API structure brings commonly used functions to the top level while organizing specialized functions into logical categories.

```mermaid
flowchart TD
    root[ember_ml] --> tensors["array(), ones(), zeros(), etc."]
    root --> basic_ops["add(), matmul(), reshape(), etc."]
    root --> categories["Categorized Operations"]
    root --> nn["nn (modules)"]
    
    categories --> linalg["em.linalg.svd(), etc."]
    categories --> stats["em.stats.mean(), etc."]
    categories --> random["em.random.normal(), etc."]
    categories --> activations["em.activations.relu(), etc."]
    
    nn --> modules["Sequential, Conv2D, etc."]
    nn --> activations_classes["ReLU, Sigmoid classes"]
    
    classDef topLevel fill:#9f6,stroke:#333,stroke-width:2px;
    classDef categories fill:#69f,stroke:#333,stroke-width:2px;
    classDef modules fill:#f96,stroke:#333,stroke-width:2px;
    
    class tensors,basic_ops topLevel;
    class categories,linalg,stats,random,activations categories;
    class nn,modules,activations_classes modules;
```

## Import Pattern Comparison

### Current Import Pattern:
```python
# Multiple import locations
from ember_ml import tensor
from ember_ml import ops

# Mixed operation locations
x = tensor.array([1, 2, 3])  # tensor creation
y = ops.add(x, x)            # ops module
z = tensor.reshape(y, (3,))  # tensor module again
```

### New Import Pattern:
```python
# Simple top-level import
import ember_ml as em

# Consistent operation location
x = em.array([1, 2, 3])      # top level
y = em.add(x, x)             # top level
z = em.reshape(y, (3,))      # top level

# Specialized operations with clear categorization
w = em.linalg.svd(z)         # categorized
```

## Backend System

The backend system will maintain the same functionality while supporting the new API structure:

```mermaid
flowchart LR
    api["API Layer<br>(em.array, em.add, etc.)"] --> registry["Registry System"]
    registry --> backends["Backend Selection"]
    backends --> numpy["NumPy Backend"]
    backends --> torch["PyTorch Backend"]
    backends --> mlx["MLX Backend"]
    
    classDef api fill:#9f6,stroke:#333,stroke-width:2px;
    classDef system fill:#69f,stroke:#333,stroke-width:2px;
    classDef backends fill:#f96,stroke:#333,stroke-width:2px;
    
    class api api;
    class registry,backends system;
    class numpy,torch,mlx backends;
```

## Implementation Strategy

```mermaid
gantt
    title API Reorganization Implementation Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %m-%d
    
    section Phase 1
    Create directory structure        :a1, 2025-06-24, 2d
    
    section Phase 2
    Move tensor operations            :a2, 2025-06-26, 3d
    
    section Phase 3
    Update backend implementation     :a3, 2025-06-29, 4d
    
    section Phase 4
    Integrate activation functions    :a4, 2025-07-03, 2d
    
    section Phase 5
    Testing                           :a5, 2025-07-05, 3d
    Documentation                     :a6, 2025-07-08, 3d
    
    section Phase 6
    Backward compatibility            :a7, 2025-07-11, 3d
```

This visualization provides a clear comparison between the current and proposed API structures, showing how the reorganization will create a more intuitive and consistent user experience.
