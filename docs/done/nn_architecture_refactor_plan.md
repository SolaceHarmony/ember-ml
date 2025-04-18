# Ember ML Neural Network (nn) Architecture Refactoring Plan (Version 3 - Incorporating Deferred Build)

## 1. Introduction

**Goal:** To refactor the `ember_ml.nn` package to establish a clear, consistent, extensible, and maintainable architecture for neural network components (layers, cells, activations, etc.). This plan aims to align with common practices in modern ML frameworks (Keras, TensorFlow) and the original NCP library's design principles, while supporting Ember ML's unique features like backend agnosticism and specialized recurrent cells (LTC, CfC, NCP). **A key decision is the adoption of a deferred build mechanism for enhanced flexibility.**

**Rationale:** The previous structure had inconsistencies in hierarchy (Layer vs. Cell), redundant abstractions (Activations, Wiring interfaces), unclear component responsibilities, and followed a "build-at-init" pattern inconsistent with modern frameworks and even the TensorFlow version of the original `ncps` library. This refactoring addresses these issues for improved clarity, maintainability, flexibility, and extensibility.

## 2. Core Concepts & Class Hierarchy

The foundation relies on a clear hierarchy based on `Module`, mimicking established frameworks and incorporating deferred initialization:

*   **`nn.modules.Module` (`BaseModule`):** The universal base class for all neural network components.
    *   Manages parameters (`nn.modules.Parameter`), submodules, buffers, training mode.
    *   Defines the core `forward` pass interface (to be overridden).
    *   **Deferred Build:** Implements the core deferred build logic. Contains a `built` flag (initially `False`), an empty `build(self, input_shape)` method for subclasses to override, and modifies `__call__` to invoke `build` once before the first `forward` call using the runtime input shape.
    *   Includes base serialization methods (`get_config`, `from_config`).

*   **Layers (e.g., `Dense`, `LTC`, `CfC`, `LSTM`, `BatchNormalization`)**: Inherit directly from `Module`. Represent self-contained computational units processing entire inputs or sequences. Reside primarily in `nn.modules/` or specialized subdirs like `nn/modules/rnn/`. They override `build` to create input-dependent weights.

    *   **RNN Layers (`LTC`, `CfC`, `LSTM`, etc.):** A specific type of Layer that internally manages sequence iteration (handling `batch_first`, `return_sequences`, `go_backwards`) and contains a corresponding `Cell` instance to perform the single-timestep computation. Its `build` method ensures the internal cell is also built.

*   **`nn.modules.ModuleCell`**: Inherits `Module`. A specialized base class for single-step, stateful computations (standard RNN cells).
    *   Takes `input_size`, `hidden_size` in `__init__`.
    *   Defines default `state_size`, `output_size` properties (often based on `hidden_size`).
    *   Overrides `build` to create input-dependent weights (like input kernels).
    *   Defines a `forward(input_t, state_prev)` signature (to be overridden).

*   **`nn.modules.ModuleWiredCell`**: Inherits `ModuleCell`. Specialization for cells whose structure is defined by a `NeuronMap`.
    *   Takes `NeuronMap` instance in `__init__` (does not take `input_size` or `hidden_size`).
    *   Overrides `build(self, input_shape)`: Extracts `input_dim`, calls `neuron_map.build(input_dim)`, sets `self.input_size`, `self.hidden_size` (from `map.units`), `self.output_size` (from `map.output_dim`). Calls `super().build(input_shape)`.
    *   Implements specific `get_config`/`from_config` for nested `NeuronMap` serialization.

*   **Concrete Cells (e.g., `LTCCell`, `WiredCfCCell`, `LSTMCell`, `GRUCell`, `RNNCell`)**: Inherit from `ModuleCell` or `ModuleWiredCell`.
    *   Implement the specific mathematical logic for a single computational step in `forward`.
    *   Override `build(self, input_shape)`: Call `super().build(input_shape)` first, then create cell-specific parameters using dimensions set during the parent `build` process.
    *   Implement specific `get_config`/`from_config`. Reside in `nn/modules/rnn/` or similar.

## 3. NeuronMap (formerly Wiring)

*   **Concept:** Renamed from `Wiring` for clarity.
*   **Role:** A **required structural blueprint** for wired cells/layers. It defines `units`, `input_dim` (deferred), `output_dim`, and the internal connectivity graph (adjacency matrices, masks, polarity, sparsity rules). It is a configuration/definition object, *not* a computational module. It encapsulates the *how* of connectivity generation. **Can be extended to include spatial information.**
*   **Location:** Base `NeuronMap` and implementations (`FullyConnectedMap`, `NCPMap`, `RandomMap`, future spatial maps) reside in `ember_ml/nn/modules/wiring/`.
*   **Constructor Requirement:** Wired Cells (`LTCCell`, `WiredCfCCell`) **must** accept a `NeuronMap` instance in `__init__`. Layers using wired cells (`LTC`, wired `CfC`) also accept a `NeuronMap`. Dimensions like `units`, `output_dim`, and eventually `input_dim` are derived *from the map* during the build process.
*   **`build(input_dim)` Method:** Handles finalization of input-dependent structures (e.g., `sensory_adjacency_matrix`). **Called by the owning Layer or Cell during its deferred `build` phase**, receiving the runtime `input_dim`. Subclasses must implement this and set `self._built = True`.
*   **Serialization:** `get_config` saves constructor arguments and potentially the *current* state (like `input_dim` if built). `from_config` reconstructs the object.

## 4. Deferred Build Pattern

*   **Rationale:** Adopted for consistency with modern frameworks (Keras, TF), alignment with `ncps.tf`, increased flexibility (no need for `input_size` at init), and resolving internal architectural conflicts from partial adoption of the previous library's patterns. Diverges from the build-at-init pattern used in `ncps.torch`.
*   **Mechanism:**
    1.  `BaseModule.__call__` intercepts the first execution.
    2.  It determines the `input_shape` from the input arguments.
    3.  It calls `self.build(input_shape)`.
    4.  The `build` method propagates down the inheritance chain (`BaseModule` -> `ModuleCell` -> `ModuleWiredCell` -> Concrete Cell).
    5.  `ModuleWiredCell.build` calls `neuron_map.build(input_dim)`, finalizing the map.
    6.  Each class in the chain creates its input-dependent weights/parameters within its `build` override, using dimensions finalized by parent `build` calls or the built map.
    7.  `BaseModule.__call__` sets `self.built = True` and proceeds with `self.forward`.

## 5. Activations

*   **Concept:** Composable `Module` subclasses, wrapping `ops` functions.
*   **Implementation:** Classes (`ReLU`, `Tanh`, etc.) inherit `nn.modules.Module`. `forward` calls the corresponding `ops` function. **No `build` method needed** as they are typically input-shape independent.
*   **Location:** Reside in `ember_ml/nn/modules/activations/`.
*   **Serialization:** Simple activations rely on base `Module` methods. Activations with parameters (`Softmax`, `Dropout`) implement `get_config`.

## 6. Initializers

*   **Concept:** Remain utility *functions* for `Parameter` creation.
*   **Location:** Remain in `ember_ml/nn/initializers/`.

## 7. Containers (`Sequential`)

*   **Concept:** `nn.container.Sequential` (inherits `Module`) executes child `Module`s linearly.
*   **Compatibility:** Compatible with Layers and Activation Modules inheriting `Module`. Handles deferred build implicitly by calling children sequentially.
*   **Serialization:** Implements `get_config`/`from_config` to handle nested module serialization.

## 8. Architecture Diagram (Conceptual - Reflecting Deferred Build)

```mermaid
graph TD
    subgraph EmberML_NN_Modules [ember_ml.nn.modules]
        direction TB
        BaseModule[Module built: bool, build()]
        BaseCell[ModuleCell inherits Module]
        BaseWiredCell[ModuleWiredCell inherits ModuleCell<br/>build() calls map.build()]

        subgraph EmberML_RNN [rnn]
            direction TB
            LTCCell[LTCCell inherits ModuleWiredCell<br/>build() allocates params]
            CfCCell[CfCCell inherits ModuleCell<br/>build() allocates params]
            LSTMCell[LSTMCell inherits ModuleCell<br/>build() allocates params]
            GRUCell[GRUCell inherits ModuleCell<br/>build() allocates params]
            RNNCell[RNNCell inherits ModuleCell<br/>build() allocates params]
        end

        subgraph EmberML_Wiring [wiring]
             NeuronMap[NeuronMap _built: bool, build()]
             NCPMap[NCPMap inherits NeuronMap]
             FullyConnectedMap[FullyConnectedMap inherits NeuronMap]
             RandomMap[RandomMap inherits NeuronMap]
             SpatialMap[SpatialNeuronMap inherits NeuronMap Future]
        end

        subgraph EmberML_Activations [activations]
             Activation[Activation inherits Module]
        end
    end

    BaseCell --> BaseModule
    BaseWiredCell --> BaseCell
    NCPMap --> NeuronMap
    FullyConnectedMap --> NeuronMap
    RandomMap --> NeuronMap
    SpatialMap --> NeuronMap

    LTCCell --> BaseWiredCell
    CfCCell --> BaseCell
    LSTMCell --> BaseCell
    GRUCell --> BaseCell
    RNNCell --> BaseCell
    Activation --> BaseModule

    BaseWiredCell -- uses --> NeuronMap

    classDef base fill:#D8BFD8,stroke:#333,stroke-width:2px;
    classDef cellbase fill:#ADD8E6,stroke:#333,stroke-width:1px;
    classDef wiredbase fill:#87CEEB,stroke:#333,stroke-width:1px;
    classDef cell fill:#B0E0E6,stroke:#333,stroke-width:1px;
    classDef wiring fill:#lightgrey,stroke:#333,stroke-width:1px;
    classDef activation fill:#FFDAB9,stroke:#333,stroke-width:1px;


    class BaseModule base;
    class BaseCell cellbase;
    class BaseWiredCell wiredbase;
    class LTCCell,CfCCell,LSTMCell,GRUCell,RNNCell cell;
    class NeuronMap,NCPMap,FullyConnectedMap,RandomMap,SpatialMap wiring;
    class Activation activation;
```

## 9. Usage Examples (Reflecting Deferred Build)

**Example A: Feed-Forward**
```python
from ember_ml.nn.container import Sequential
from ember_ml.nn.modules import Dense
from ember_ml.nn.modules.activations import ReLU

model = Sequential([
    Dense(units=128), # input_size is NOT needed here
    ReLU(),
    Dense(units=10) # input_size inferred from previous layer
])
# Build happens implicitly on first call: model(input_tensor)
```

**Example B: Wired LTC Layer**
```python
from ember_ml.nn.modules.rnn import LTC
from ember_ml.nn.modules.wiring import NCPMap # Use specific map

# 1. Explicitly create the NeuronMap blueprint
# Note: No input_dim needed here
ncp_map = NCPMap(inter_neurons=40, motor_neurons=10, sparsity_level=0.5)

# 2. Pass map to the Layer constructor. Layer doesn't need input_size.
ltc_layer = LTC(neuron_map_or_cell=ncp_map)

model = Sequential([ ltc_layer ])
# Build happens implicitly on first call: model(input_tensor)
# Layer's build will call map's build with actual input dim
```

## 10. Phased Implementation Plan & Status

**Phase 1: NeuronMap Renaming & Consolidation (Complete)**
*   Renamed `Wiring` -> `NeuronMap`, implementations -> `*Map`.
*   Consolidated map classes into `nn/modules/wiring/`.
*   Updated `__init__` files.
*   Updated imports across codebase (modules, examples, tests).
*   Refactored Layer/Cell constructors to require `NeuronMap` where appropriate.
*   Deleted old `nn/wirings/` directory.

**Phase 2: Activation Module Implementation (Complete)**
*   Created `nn/modules/activations/`.
*   Implemented `ReLU`, `Tanh`, `Sigmoid`, `Softmax`, `Softplus`, `LeCunTanh`, `Dropout` modules.
*   Created/Updated `__init__` files.
*   Deleted old `nn/activations/` directory.

**Phase 3: Deferred Build Refactor & Hierarchy Fixes (Current Focus)**
*   **Goal:** Implement deferred build mechanism in base classes and refactor wired cells. Fix CfC inconsistencies.
*   **Plan:** Follow detailed steps outlined in `docs/plans/rnn_deferred_build_refactor.md`.
*   **Status:** Plan documented and awaiting implementation.

**Phase 4: Spatial NeuronMap Integration (Future)**
*   **Goal:** Extend `NeuronMap` to support spatial coordinates and distance-based connectivity rules using backend-agnostic ops.
*   **Approach:** Add optional coordinate attributes/handling to `NeuronMap`. Create `SpatialNeuronMap` base and concrete subclasses (e.g., `SpatiallyRestrictedRandomMap`) that override `build` to incorporate spatial logic. Ensure backend purity.

**Phase 5: Testing & Documentation (Ongoing)**
*   Run full test suite (`pytest tests/`) to catch remaining errors and validate serialization after Phase 3.
*   Add specific tests for Deferred Build behavior, Spatial Maps (when implemented).
*   Update all relevant docstrings and API documentation (`.md` files) to reflect the final structure and usage patterns.

## 11. Conclusion

This refactoring establishes a robust, clear, and extensible foundation for `ember_ml.nn`, aligned with modern framework practices (deferred build) and the original NCP library's TensorFlow implementation. It simplifies activations, clarifies the Layer/Cell/NeuronMap roles, standardizes initialization, and provides a consistent structure for future development, including spatial embedding concepts.