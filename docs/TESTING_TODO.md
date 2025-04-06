# Ember ML - Remaining Test Suite Tasks (as of 2025-04-04 ~3:40 PM PT)

This document summarizes the remaining tasks to achieve comprehensive and accurate test coverage for the Ember ML library, based on the `pytest` output after addressing major structural and refactoring issues.

## I. Missing Implementations / API Exposure

1.  **`ops.dropout`:**
    *   **Issue:** `AttributeError: module 'ember_ml.ops' has no attribute 'dropout'`. The `Dropout` module requires this.
    *   **Fix:** Define `dropout` in an appropriate ops interface, add dispatcher to `ops/__init__.py`, implement in all backends (`numpy`, `torch`, `mlx`). Test in `test_nn_activations.py`.
2.  **`ops.maximum`:**
    *   **Issue:** `AttributeError: module 'ember_ml.ops' has no attribute 'maximum'`. Needed for `relu` test calculation.
    *   **Fix:** Define `maximum` in `ops.interfaces.math_ops`, add dispatcher, implement in backends. Update `test_ops_math.py`.
3.  **`nn.features.PCA`:**
    *   **Issue:** `AttributeError: module 'ember_ml.nn.features' has no attribute 'PCA'`.
    *   **Fix:** Ensure `PCA` class from `common` is imported and added to `__all__` in `ember_ml/nn/features/__init__.py`.
4.  **`nn.features` Backend/Common Classes (`TensorFeatures`):**
    *   **Issue:** `AttributeError: module ... has no attribute '{BackendPrefix}TensorFeatures'` or `AttributeError: module 'ember_ml.nn.features.common' has no attribute 'TensorFeatures'`. The dynamic loading/fallback fails.
    *   **Fix:** Implement the necessary `TensorFeatures` classes (e.g., `NumpyTensorFeatures`, `TorchTensorFeatures`, `MLXTensorFeatures`) in the respective backend feature ops files, or create/fix a common implementation if applicable. Update `nn/features/__init__.py` loading logic if needed.
5.  **`LossOps` Backend Implementations:**
    *   **Issue:** `AttributeError: module 'ember_ml.backend.*' has no attribute '{BackendPrefix}LossOps'`. Although the fix in `ops/__init__.py` was applied, these errors persist, suggesting the *classes themselves* might still be missing or misnamed in the backend files.
    *   **Fix:** Double-check existence and naming (`NumpyLossOps`, `TorchLossOps`, `MLXLossOps`) in `ember_ml/backend/*/loss_ops.py`. Create/rename as necessary.
6.  **`VectorOps` Backend Implementations:**
    *   **Issue:** `AttributeError: module 'ember_ml.backend.*' has no attribute '{BackendPrefix}VectorOps'`.
    *   **Fix:** Double-check existence and naming (`NumpyVectorOps`, `TorchVectorOps`, `MLXVectorOps`) in `ember_ml/backend/*/vector_ops.py`. Create/rename as necessary.

## II. Module Initialization / Signature Errors

1.  **Neuron Maps (`NCPMap`, `FullyConnectedMap`, `RandomMap`):**
    *   **Issue:** `TypeError: ... __init__() got an unexpected keyword argument ...` (`command_neurons`, `output_size`).
    *   **Fix:** Correct the arguments passed during instantiation in `test_nn_wiring.py` and `test_nn_rnn.py` (for `LTCCell` test) to match the actual `__init__` signatures of these map classes.
2.  **`AutoNCP`:**
    *   **Issue:** Broadcasting errors (`ValueError`, `RuntimeError`).
    *   **Fix:** Debug the internal forward pass logic of `AutoNCP`, likely related to shape mismatches during matrix operations.
3.  **RNN Modules (`RNNCell`, `LSTMCell`, `GRUCell`, `CfCCell`, `LTCCell`, Layers):**
    *   **Issue:** Various `TypeError`s (invalid args, missing args like `cell_or_map` for `CfC`), `AttributeError` (`glorot_uniform`, `.units`).
    *   **Fix:** Systematically review and correct the `__init__` signatures and internal initialization logic (e.g., weight initialization, attribute setting) for all basic RNN cells and layers. Ensure they align with the base module expectations and API documentation.
4.  **`Dense` Module:**
    *   **Issue:** `AssertionError: Dense output is not EmberTensor`.
    *   **Fix:** Ensure the `Dense.forward` method explicitly wraps its final result in `EmberTensor()`.

## III. Core Tensor / DType Issues

1.  **Return Types (`EmberTensor`):**
    *   **Issue:** Creation (`zeros`, `ones`, `eye`) and random (`random_*`) functions fail `isinstance(..., EmberTensor)` checks.
    *   **Fix:** Ensure the common tensor functions (`tensor.zeros`, etc.) and their backend implementations consistently wrap the result in an `EmberTensor` instance before returning.
2.  **`.dtype` Property:**
    *   **Issue:** `AssertionError: Dtype property mismatch...`. The `.dtype` attribute returns backend-specific dtypes.
    *   **Fix:** Modify the `EmberTensor.dtype` property (likely in `ember_ml/nn/tensor/common/ember_tensor.py`) to return the stored `EmberDType` object, not the underlying backend dtype.
3.  **`EmberDType` Handling:**
    *   **Issue:** `TypeError: unhashable type: 'EmberDType'`, `ValueError: Cannot convert ... to EmberDType`.
    *   **Fix:** Implement correct `__hash__` and `__eq__` methods for the `EmberDType` class. Fix `to_dtype_str` and `from_dtype_str` logic to handle conversions properly, especially for backend-specific dtype objects. Update tests in `test_nn_tensor_dtype.py`.

## IV. Specific Op Implementation Errors

1.  **`tensor.scatter`:**
    *   **Issue:** `TypeError` related to argument types/shapes across backends.
    *   **Fix:** Debug and correct the implementation in `ember_ml/backend/*/tensor/ops/indexing.py`.
2.  **`tensor.tensor_scatter_nd_update[numpy]`:**
    *   **Issue:** `TypeError: isinstance() arg 2 must be a type...`.
    *   **Fix:** Correct the type check within the NumPy implementation.
3.  **`tensor.slice`:**
    *   **Issue:** `TypeError: slice() got unexpected keyword argument 'begin'`.
    *   **Fix:** Correct the function signature in common/backend implementations to accept `begin` and `size`. Update tests.
4.  **`tensor.pad`:**
    *   **Issue:** `TypeError: ...pad() got unexpected keyword argument 'mode'`.
    *   **Fix:** Correct function signature in common/backend implementations to accept `mode` and `constant_values`. Update tests.
5.  **`tensor.transpose`:**
    *   **Issue:** Fails for >2D tensors (Torch `t()`), shape assertion errors (Numpy/MLX).
    *   **Fix:** Implement robust transpose logic for N-dimensions using `permute` or equivalent in backends.
6.  **`tensor.tile[torch]`:**
    *   **Issue:** `RuntimeError: Long did not match Float`.
    *   **Fix:** Ensure dtype consistency within the PyTorch `tile` implementation.
7.  **`ops.stats.*` Backend Conversion/Signature Errors:**
    *   **Issue:** `ValueError: Cannot convert <class 'torch.Tensor'> to NumPy array`, `TypeError: sum() received an invalid combination...`, `TypeError: min() received...`, `RuntimeError: Float did not match Long` (median), `AxisError` (percentile).
    *   **Fix:** Primarily fix the `_convert_input` issue (Task 6 above). If errors persist, debug the specific stats function wrappers/implementations in `ember_ml/backend/numpy/stats/` to handle arguments/types correctly.
8.  **`ops.vector.*[mlx]` Signature Errors:**
    *   **Issue:** `TypeError` for `cosine_similarity`, `exponential_decay`, `gaussian` related to missing/unexpected keyword arguments.
    *   **Fix:** Correct the function signatures or the calling lambda functions in `ember_ml/ops/__init__.py` and `ember_ml/backend/mlx/vector_ops.py`.
9.  **`ops.isnan`:**
    *   **Issue:** `NameError: name 't1' is not defined`.
    *   **Fix:** Correct the variable name used within the `test_ops_isnan` function in `tests/test_ops_comparison.py`.
10. **`ops.io.save`:**
    *   **Issue:** `TypeError: expected str... not EmberTensor`, missing `allow_pickle` for MLX.
    *   **Fix:** Correct the `save` function signature/implementation in backends and `ops/io_ops.py` and update tests.
11. **MLX Device Issues:**
    *   **Issue:** `AssertionError` in `test_get_available_devices`, `test_to_device_mps`, `test_get_device`.
    *   **Fix:** Investigate and correct device reporting and handling in `ember_ml/backend/mlx/device_ops.py`.
12. **`lecun_tanh` Precision:**
    *   **Issue:** `AssertionError` due to small float differences.
    *   **Fix:** Increase `atol` in `ops.allclose` for this specific test in `test_nn_activations.py` or investigate calculation differences.

## V. Test Logic Errors

1.  **`test_ops_linearalg.py::test_linearalg_eigvals`:**
    *   **Issue:** `NameError: name 'mat_a' is not defined`.
    *   **Fix:** Correct the test logic to use the defined symmetric matrix.

This list provides a structured approach to tackling the remaining test failures.