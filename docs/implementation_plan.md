# Implementation Plan for EmberHarmony Control Theory Integration

## Overview

This document outlines the plan for integrating control theory experiments into the EmberHarmony framework. The plan addresses the gaps identified in the initial implementation attempt, particularly around optimizers, backend integration, and proper use of the EmberHarmony ops module.

## Current Status

The initial implementation attempt revealed several issues:

1. **Missing Optimizer Implementation**: The code assumed the existence of `emberharmony.training.Optimizer` and `emberharmony.training.Loss` classes that don't exist yet.
2. **Improper Backend Usage**: The implementation didn't properly leverage EmberHarmony's backend system.
3. **Direct Framework Usage**: The code directly used PyTorch/TensorFlow constructs instead of EmberHarmony's ops abstraction.
4. **Assumptions About Shape Functions**: The code made assumptions about shape functions without checking their implementation.

## Implementation Plan

### Phase 1: Research and Analysis

1. **Examine Existing Code**:
   - Review the EmberHarmony backend system to understand how it selects and overrides backend implementations
   - Study the ops module to understand how it provides a unified interface across backends
   - Analyze existing loss function implementations to understand the pattern for backend-agnostic implementations

2. **Identify Open Source Implementations**:
   - Look for CfC implementations in site-packages (particularly in ncps.torch) using CLI tools
   - Examine optimizer implementations in PyTorch, TensorFlow, and MLX
   - Study how other libraries implement backend-agnostic optimizers
   - Use CLI commands to examine the structure and organization of ncps.torch

3. **Study Cell Type Examples**:
   - Examine `cell_type_examples.ipynb` to understand different cell types
   - Identify all cell types that need to be implemented in EmberHarmony
   - Research xLSTM implementation using Browser and GitHub repositories
   - Understand how these cell types can be implemented using only EmberHarmony ops

### Phase 2: Core Infrastructure Development

1. **Create Optimizer Module Structure**:
   - Implement `emberharmony.nn.optimizers/__init__.py` to define the module structure
   - Create a base `optimizer.py` file with the base `Optimizer` class that defines the interface for all optimizers
   - Implement backend-specific base classes in separate files (e.g., `torch_optimizer.py`, `mlx_optimizer.py`)

2. **Implement Common Optimizers**:
   - Implement SGD optimizer in `sgd.py`
   - Implement Adam optimizer in `adam.py`
   - Implement RMSprop optimizer in `rmsprop.py`
   - Create factory functions in `__init__.py` for easy creation of optimizers
   - Test each optimizer with all backends

3. **Implement Loss Functions**:
   - Ensure Loss functions are properly implemented and accessible
   - Create factory functions for easy creation of loss functions
   - Test loss functions with all backends

### Phase 3: CfC Implementation

1. **Examine ncps.torch CfC Implementation Structure**:
   - Use CLI tools to examine the structure of ncps.torch
   - Identify all CfC-related classes and their relationships
   - Understand how the classes are organized and separated
   - Plan the file structure for the EmberHarmony implementation

2. **Implement Base CfC Components**:
   - Create `emberharmony/nn/cfc/base_cell.py` with base cell implementation
   - Create `emberharmony/nn/cfc/cfc_cell.py` with CfCCell implementation
   - Create `emberharmony/nn/cfc/ltc_cell.py` with LTCCell implementation (if present in ncps.torch)
   - Ensure all implementations use ops functions
   - Test each component with different backends

3. **Implement Wired CfC Components**:
   - Create `emberharmony/nn/cfc/wired_cell.py` with base wired cell implementation
   - Create `emberharmony/nn/cfc/wired_cfc_cell.py` with WiredCfCCell implementation
   - Create `emberharmony/nn/cfc/wired_ltc_cell.py` with WiredLTCCell implementation (if present in ncps.torch)
   - Ensure all implementations use ops functions and work with the wiring system
   - Test each component with different backends and wiring configurations

4. **Implement CfC Layers**:
   - Create `emberharmony/nn/cfc/cfc_layer.py` with CfC layer implementation
   - Create `emberharmony/nn/cfc/ltc_layer.py` with LTC layer implementation (if present in ncps.torch)
   - Create `emberharmony/nn/cfc/mixed_memory.py` with mixed memory implementation
   - Ensure all implementations use ops functions
   - Test each layer with different backends

5. **Update CfC Module Structure**:
   - Update `emberharmony/nn/cfc/__init__.py` to export all CfC components
   - Create `emberharmony/nn/cfc/README.md` with documentation
   - Ensure all components are properly exported
   - Test the module structure by importing components

### Phase 4: Additional Cell Types Implementation

1. **Implement Basic Cell Types**:
   - Create `emberharmony/nn/cells/__init__.py` to define the module structure
   - Implement LSTM cell in `emberharmony/nn/cells/lstm.py`
   - Implement GRU cell in `emberharmony/nn/cells/gru.py`
   - Implement RNN cell in `emberharmony/nn/cells/rnn.py`
   - Ensure all implementations use only EmberHarmony ops

2. **Implement Advanced Cell Types**:
   - Implement xLSTM cell in `emberharmony/nn/cells/xlstm.py` based on research
   - Implement any other advanced cell types from `cell_type_examples.ipynb`
   - Ensure all implementations use only EmberHarmony ops
   - Extend EmberHarmony as needed to support these cell types

3. **Create Cell Type Layers**:
   - Implement layer wrappers for each cell type
   - Create factory functions for easy creation of cell layers
   - Ensure all implementations use only EmberHarmony ops

4. **Test Cell Type Implementations**:
   - Create tests for each cell type
   - Verify that implementations match the behavior in `cell_type_examples.ipynb`
   - Test with different backends to ensure backend agnosticism

### Phase 5: Control Theory Integration

1. **Update Control Theory Experiments**:
   - Refactor `liquid_control_experiments_emberharmony.py` to use the new optimizer implementation
   - Ensure all tensor operations use ops functions
   - Fix any shape-related issues
   - Test the updated experiments with each backend
   - Verify that the experiments work correctly with all backends

2. **Create Example**:
   - Update `examples/cfc_example.py` to use the new implementations
   - Create examples for other cell types in `examples/cell_types_example.py`
   - Ensure examples demonstrate all capabilities
   - Test examples with different backends
   - Make them good references for users
   - Verify that the examples work correctly with all backends

### Phase 6: Testing and Documentation

1. **Write Tests for Optimizers**:
   - Create tests for base Optimizer class in `tests/test_optimizers_base.py`
   - Create tests for SGD optimizer in `tests/test_optimizers_sgd.py`
   - Create tests for Adam optimizer in `tests/test_optimizers_adam.py`
   - Create tests for RMSprop optimizer in `tests/test_optimizers_rmsprop.py`
   - Ensure tests cover all backends
   - Reference existing tests in the `./testing` folder for examples

2. **Write Tests for CfC Implementation**:
   - Create tests for CfCCell in `tests/test_cfc_cell.py`
   - Create tests for WiredCfCCell in `tests/test_wired_cfc_cell.py`
   - Create tests for CfC layer in `tests/test_cfc_layer.py`
   - Create tests for CfC with different wirings in `tests/test_cfc_wiring.py`
   - Ensure tests cover all backends
   - Reference existing tests in the `./testing` folder for examples

3. **Write Tests for Cell Types**:
   - Create tests for LSTM cell in `tests/test_lstm_cell.py`
   - Create tests for GRU cell in `tests/test_gru_cell.py`
   - Create tests for RNN cell in `tests/test_rnn_cell.py`
   - Create tests for xLSTM cell in `tests/test_xlstm_cell.py`
   - Create tests for other cell types
   - Ensure tests cover all backends
   - Reference existing tests in the `./testing` folder for examples

4. **Write Integration Tests**:
   - Create tests for control theory experiments in `tests/test_control_theory.py`
   - Create tests for CfC example in `tests/test_cfc_example.py`
   - Create tests for cell types example in `tests/test_cell_types_example.py`
   - Create tests for CfC with optimizers in `tests/test_cfc_optimizers.py`
   - Ensure tests cover all backends
   - Reference existing tests in the `./testing` folder for examples

5. **Test Backend Compatibility**:
   - Create tests specifically for backend compatibility in `tests/test_backend_compatibility.py`
   - Test switching between backends at runtime
   - Test that all components work with all backends
   - Verify that the same code works correctly with different backends
   - Reference existing tests in the `./testing` folder for examples

6. **Update Documentation**:
   - Document optimizer API
   - Document CfC implementation
   - Document cell types implementation
   - Update control theory documentation
   - Create usage examples
   - Ensure documentation covers all backends

## Implementation Details

### Optimizer Implementation

The optimizer implementation will follow these principles:

1. **Backend Agnosticism**: The optimizer will work with any backend (PyTorch, TensorFlow, MLX)
2. **Unified Interface**: The optimizer will provide a consistent interface regardless of the backend
3. **Factory Functions**: Factory functions will make it easy to create optimizers
4. **Modular Structure**: Each optimizer will be in its own file for better maintainability

#### File Structure

```
emberharmony/
  nn/
    optimizers/
      __init__.py          # Factory functions and exports
      optimizer.py         # Base Optimizer class
      sgd.py               # SGD optimizer
      adam.py              # Adam optimizer
      rmsprop.py           # RMSprop optimizer
      adagrad.py           # Adagrad optimizer
      adadelta.py          # Adadelta optimizer
      backends/
        torch_optimizer.py # PyTorch-specific base class
        numpy_optimizer.py # NumPy-specific base class
        mlx_optimizer.py   # MLX-specific base class
```

#### Example Usage

```python
# Example usage
from emberharmony.nn import optimizers

# Create an Adam optimizer
optimizer = optimizers.adam(
    parameters=model.parameters(),
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0
)

# Zero gradients
optimizer.zero_grad()

# Compute gradients
grads = ops.gradients(loss, model.parameters())

# Update parameters
optimizer.step(grads)
```

### CfC Implementation

The CfC implementation will be ported from ncps.torch and adapted to work with EmberHarmony:

1. **Use ops Functions**: All tensor operations will use ops functions
2. **Backend Agnosticism**: The implementation will work with any backend
3. **Wiring Integration**: The implementation will work with EmberHarmony's wiring system
4. **Modular Structure**: Each class will be in its own file, following the same pattern as ncps.torch

#### File Structure

```
emberharmony/
  nn/
    cfc/
      __init__.py          # Exports and factory functions
      base_cell.py         # Base cell implementation
      cfc_cell.py          # CfCCell implementation
      ltc_cell.py          # LTCCell implementation (if present in ncps.torch)
      wired_cell.py        # Base wired cell implementation
      wired_cfc_cell.py    # WiredCfCCell implementation
      wired_ltc_cell.py    # WiredLTCCell implementation (if present in ncps.torch)
      cfc_layer.py         # CfC layer implementation
      ltc_layer.py         # LTC layer implementation (if present in ncps.torch)
      mixed_memory.py      # Mixed memory implementation
      README.md            # Documentation
```

### Cell Types Implementation

The cell types implementation will be based on `cell_type_examples.ipynb` and adapted to work with EmberHarmony:

1. **Use ops Functions**: All tensor operations will use ops functions
2. **Backend Agnosticism**: The implementation will work with any backend
3. **EmberHarmony Only**: No direct use of numpy, torch, or keras
4. **Modular Structure**: Each cell type will be in its own file

#### File Structure

```
emberharmony/
  nn/
    cells/
      __init__.py          # Exports and factory functions
      base_cell.py         # Base cell implementation
      lstm.py              # LSTM cell implementation
      gru.py               # GRU cell implementation
      rnn.py               # RNN cell implementation
      xlstm.py             # xLSTM cell implementation
      other_cells.py       # Other cell types from cell_type_examples.ipynb
      README.md            # Documentation
```

#### Example Usage

```python
# Example usage
from emberharmony.nn.cells import LSTM, GRU, RNN, xLSTM

# Create an LSTM cell
lstm_cell = LSTM(
    input_size=10,
    hidden_size=20,
    activation="tanh",
    recurrent_activation="sigmoid"
)

# Create an xLSTM cell
xlstm_cell = xLSTM(
    input_size=10,
    hidden_size=20,
    activation="tanh",
    recurrent_activation="sigmoid"
)
```

## Important Reminders for Implementation

1. **Use CLI Tools**: Always use CLI tools to check implementations in site-packages. This will help you understand the structure and organization of the original code.

2. **Separate Classes**: Follow the same pattern as ncps.torch and separate different classes into different files. This makes the code more maintainable and easier to understand.

3. **Test Thoroughly**: Test each component thoroughly, especially with different backends. Make sure that the same code works correctly with PyTorch, NumPy, and MLX backends.

4. **Reference Existing Tests**: Look at the existing tests in the `./testing` folder for examples of how to write tests for EmberHarmony components.

5. **Backend Agnosticism**: Always use ops functions instead of direct backend functions. This ensures that the code works with all backends.

6. **Documentation**: Document all components thoroughly, including examples of how to use them with different backends.

7. **EmberHarmony Only**: Never use numpy, torch, or keras directly. Always use emberharmony and grow emberharmony as needed.

8. **Research xLSTM**: Use Browser and GitHub to research xLSTM implementation and ensure it's properly replicated in EmberHarmony.

## Timeline

1. **Phase 1**: 1 day
2. **Phase 2**: 2 days
3. **Phase 3**: 3 days
4. **Phase 4**: 2 days
5. **Phase 5**: 1 day
6. **Phase 6**: 2 days

Total: 11 days

## Resources

1. **ncps.torch**: Source for CfC implementation
2. **PyTorch Optimizers**: Reference for optimizer implementations
3. **TensorFlow Optimizers**: Reference for optimizer implementations
4. **MLX Optimizers**: Reference for optimizer implementations
5. **EmberHarmony Documentation**: Reference for EmberHarmony architecture and APIs
6. **./testing folder**: Reference for writing tests for EmberHarmony components
7. **cell_type_examples.ipynb**: Reference for cell types implementation
8. **GitHub repositories**: Reference for xLSTM implementation

## Conclusion

This plan outlines a comprehensive approach to integrating control theory experiments into the EmberHarmony framework. By addressing the identified gaps and following the principles of backend agnosticism and proper use of the ops module, we can create a robust and flexible implementation that leverages the full power of EmberHarmony. Additionally, by implementing various cell types from `cell_type_examples.ipynb` and researching xLSTM, we can enhance EmberHarmony's capabilities and make it a more powerful framework for neural network development.