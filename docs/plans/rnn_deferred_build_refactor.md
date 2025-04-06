# Plan: Refactor RNN Wired Cells for Deferred Build & Fix CfC

**Date:** 2025-04-05

**Goal:** Align `ModuleWiredCell` and its subclasses (e.g., `LTCCell`, `WiredCfCCell`) with the deferred initialization strategy outlined in `docs/nn_architecture_refactor_plan.md`. Additionally, fix inconsistencies found in the `CfC` layer and cell definitions.

**Problem:**
1. The current `ModuleWiredCell.__init__` attempts to build the `NeuronMap` immediately, conflicting with the deferred build goal.
2. An incorrect, duplicate `CfCCell` definition exists in `ember_ml/nn/modules/rnn/cfc.py`.
3. The `CfC` layer incorrectly instantiates `WiredCfCCell`.
4. The `CfCCell` in `cfc_cell.py` accepts unused `backbone_*` parameters.
5. The `WiredCfCCell` in `wired_cfc_cell.py` currently uses build-at-init logic, conflicting with the deferred build plan.

**Rationale for Deferred Build:**

The decision to implement deferred build, despite diverging from the original `ncps.torch` library's build-at-init pattern, is based on several factors:

1.  **Alignment with Ember ML Goals:** The explicit goal stated in `docs/nn_architecture_refactor_plan.md` is to adopt modern framework practices, including deferred initialization for flexibility.
2.  **Consistency with Modern Frameworks:** Deferred build is the standard pattern in Keras/TensorFlow, allowing layers to adapt to input shapes encountered at runtime.
3.  **Precedent in `ncps.tf`:** The original library's *own* TensorFlow implementation (`ncps.tf.LTCCell`, `ncps.tf.WiredCfCCell`) successfully utilized the deferred build pattern via Keras's base layer functionality.
4.  **Flexibility:** Deferred build removes the requirement to specify `input_size` during layer/cell initialization, making model definition less verbose and more adaptable to varying input sources.
5.  **Resolving Internal Conflict:** This refactoring directly addresses the current architectural inconsistency where `ModuleWiredCell` attempts build-at-init, conflicting with the desired deferred build pattern for its subclasses like `LTCCell`.

While build-at-init offers explicitness during initialization, deferred build provides greater flexibility and aligns better with the established patterns in the TensorFlow ecosystem and Ember ML's stated architectural direction.


**Proposed Changes:**

**Part 1: Implement Deferred Build Infrastructure**

**1. Modify `BaseModule` (`ember_ml/nn/modules/base_module.py`)**
   *   Add `self.built = False` in `__init__`.
   *   Add empty `build(self, input_shape)` method.
   *   Modify `__call__` to check `self.built`, call `self.build(input_shape)` if `False`, set `self.built = True`, then call `self.forward`.

**2. Modify `ModuleWiredCell` (`ember_ml/nn/modules/module_wired_cell.py`)**
   *   **`__init__`:** Remove `input_size` parameter and build-at-init logic (lines ~63-71). Call `super().__init__` without `input_size`.
   *   **Implement `build(self, input_shape)`:** Extract `input_dim`, call `neuron_map.build(input_dim)`, set `self.input_size`, `self.hidden_size`, `self.output_size` based on built map. Call `super().build(input_shape)`.

**3. Modify `LTCCell` (`ember_ml/nn/modules/rnn/ltc_cell.py`)**
   *   **`__init__`:** Simplify to store only `neuron_map` and LTC configs. Remove `in_features` logic and `_allocate_parameters` call. Call `super().__init__` passing only `neuron_map` and `kwargs`.
   *   **Implement `build(self, input_shape)`:** Call `super().build(input_shape)` first, then call `self._allocate_parameters()`.

**4. Modify `WiredCfCCell` (`ember_ml/nn/modules/rnn/wired_cfc_cell.py`)**
    *   **`__init__`:** Simplify to store only `neuron_map` and CfC configs. Remove `input_size` parameter and `_initialize_weights` call (line ~74). Call `super().__init__` passing only `neuron_map`, `mode`, and `kwargs`.
    *   **Implement `build(self, input_shape)`:** Call `super().build(input_shape)` first, then call `self._initialize_weights()` (the existing method can likely be reused, possibly renamed from `_initialize_wired_weights` if desired, as it correctly uses dimensions like `self.input_size` set by the parent build). Correct the recurrent mask initialization within `_initialize_weights` to use the actual map data (`self.recurrent_mask = tensor.convert_to_tensor(self.neuron_map.get_recurrent_mask())` or similar).

**Part 2: Fix CfC Layer and Cell**

**5. Delete Incorrect `CfCCell` from `cfc.py`:**
   *   Remove the entire `CfCCell` class definition from `ember_ml/nn/modules/rnn/cfc.py`.

**6. Fix `CfC` Layer (`ember_ml/nn/modules/rnn/cfc.py`)**
   *   Update `WiredCfCCell` import to use relative path: `from .wired_cfc_cell import WiredCfCCell`.
   *   Modify `WiredCfCCell` instantiation when a `NeuronMap` is passed: `self.cell = WiredCfCCell(neuron_map=cell_or_map, mixed_memory=mixed_memory, **kwargs)`. Remove `input_size=None`.

**7. Clean up `CfCCell` (`ember_ml/nn/modules/rnn/cfc_cell.py`)**
   *   Remove unused `backbone_*` parameters from `__init__` signature and attribute assignments.
   *   Update `get_config` to remove saving of these unused parameters.

**Part 3: Verify Other Components**

**8. Verify `NeuronMap` Subclasses (`ember_ml/nn/modules/wiring/`):**
    *   Ensure the `build` method in `ncp_map.py`, `fully_connected_map.py`, `random_map.py` correctly calls `self.set_input_dim(input_dim)` and sets `self._built = True`. (We added this for `ncp_map.py`).

**Rationale:** This updated plan addresses the core deferred build conflict and cleans up the identified inconsistencies in the CfC and WiredCfC implementations, leading to a more robust and internally consistent architecture aligned with the refactoring goals.

**Approval Request:** Please review this updated and more comprehensive plan. If approved, I will request to switch to Code mode to implement these changes.