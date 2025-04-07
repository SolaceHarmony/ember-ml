# Testing Multiple Backends in Ember ML

## Background

Ember ML is designed to support multiple backends (NumPy, PyTorch, MLX) through a common interface. However, testing multiple backends within the same `pytest` execution session can be problematic because:

1.  The dynamic loading of backend-specific implementations can lead to conflicts or incorrect states when switching backends between tests or test files.
2.  Tensors created with one backend are generally incompatible with operations from another backend.

## Solution: Directory-Based Backend Separation

To ensure reliable testing and complete isolation between backends, the test suite is structured into backend-specific subdirectories within `tests/`:

*   `tests/numpy_tests/`
*   `tests/torch_tests/`
*   `tests/mlx_tests/`

Each subdirectory contains copies of the relevant test files, modified to *only* run tests for that specific backend. These files use dedicated fixtures (e.g., `numpy_backend`, `torch_backend`, `mlx_backend` defined in `tests/conftest.py`) to set the correct backend before any tests in that file run.

## Usage

To run tests for a specific backend, target the corresponding directory with `pytest`:

### Run all NumPy tests:

```bash
python -m pytest tests/numpy_tests/ -v
```

### Run all PyTorch tests:

```bash
python -m pytest tests/torch_tests/ -v
```
*(Note: PyTorch tests might be skipped automatically if PyTorch is not installed)*

### Run all MLX tests:

```bash
python -m pytest tests/mlx_tests/ -v
```
*(Note: MLX tests might be skipped automatically if MLX is not installed)*

### Run a specific NumPy test file:

```bash
python -m pytest tests/numpy_tests/test_ops_math.py -v
```

### Run a specific Torch test function:

```bash
python -m pytest tests/torch_tests/test_nn_modules.py::test_dense_parameters_torch -v
```

## How it Works

1.  **Directory Structure:** Separating tests by directory ensures that when `pytest` collects tests from `tests/numpy_tests/`, only NumPy-related tests are considered for that run.
2.  **Backend Fixtures:** Each backend-specific test file uses a dedicated fixture (e.g., `numpy_backend`) from `tests/conftest.py`. This fixture sets the global Ember ML backend *before* any tests in that file execute and restores it afterward.
3.  **No Parameterization:** The parameterized `backend` fixture has been removed from `conftest.py`, eliminating the root cause of the cross-backend conflicts during a single `pytest` run.

This structure provides robust isolation and makes testing individual backends straightforward.