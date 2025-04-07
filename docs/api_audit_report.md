# Ember ML API Audit Report (Code vs. Documentation)

**Date:** 2025-04-05
**Auditor:** Thea

## 1. Introduction

This report details the findings of an audit comparing the implemented public API surface of the Ember ML library (as determined by examining `__init__.py` files and `__all__` definitions) against the documented API structure (as presented in `docs/api/index.md` and related files). The goal is to identify discrepancies and areas where the documentation may not accurately reflect the code's public interface.

Files Examined:
- `ember_ml/__init__.py`
- `ember_ml/ops/__init__.py`
- `ember_ml/nn/__init__.py`
- `docs/api/index.md`

## 2. Summary of Findings

The audit revealed several discrepancies between the implemented API and the documentation:

-   **RNN Module Exposure:** RNN-related classes are documented under `nn.modules.rnn` but are not exposed directly under the main `nn` namespace via `__all__`.
-   **Ops Structure (`linearalg`, `stats`):** The documentation presents `ops.linearalg` and `ops.stats` as distinct documented modules, while the code structure in `ops/__init__.py` suggests they might be intended for direct import or that the documentation structure needs refinement.
-   **NN Features/Activations Exposure:** `nn.features` and `nn.modules.activations` are documented as separate API sections but are not directly re-exported via `nn/__init__.py`.
-   **Top-Level Module Documentation:** Some top-level modules exposed in code (`benchmarks`, `data`, etc.) are not explicitly listed in the main API index documentation, while `initializers` is listed in docs but not exposed at the top level.
-   **Minor `__all__` Inconsistencies:** Several minor issues were found within the `__all__` list of `ops/__init__.py` (remnants, typos, duplicates).

Overall, the core `ops` and `nn` interfaces are reasonably well-aligned, but the documentation could be updated to more accurately reflect how users are expected to import specific submodules like RNNs, features, activations, linear algebra, and statistics functions.

## 3. Detailed Discrepancies

### 3.1. Modules/Functions Missing Direct Exposure in Code (vs. Docs)

-   **Recurrent Neural Networks (RNNs):**
    -   **Docs:** `docs/api/index.md` lists `ember_ml.nn.modules.rnn` containing classes like `RNN`, `LSTM`, `GRU`, `CfC`, `LTC`.
    -   **Code:** `ember_ml/nn/__init__.py` imports these classes (lines 34-36) but *does not* include them in its `__all__` list (lines 37-54).
    -   **Impact:** Users likely need to import directly from the submodule (e.g., `from ember_ml.nn.modules.rnn import LSTM`) rather than `from ember_ml.nn import LSTM`, which the documentation might imply.

-   **NN Features:**
    -   **Docs:** `docs/api/index.md` lists `ember_ml.nn.features` as a distinct API section.
    -   **Code:** The `ember_ml.nn.features` module exists, but its contents are not re-exported via `ember_ml/nn/__init__.py`'s `__all__`.
    -   **Impact:** Users need to import directly from `ember_ml.nn.features`.

-   **NN Activations:**
    -   **Docs:** `docs/api/index.md` lists `ember_ml.nn.modules.activations`.
    -   **Code:** The `ember_ml.nn.modules.activations` module exists, but its contents are not re-exported via `ember_ml/nn/__init__.py`'s `__all__`. `ops/__init__.py` *does* expose some activation *functions* directly (relu, sigmoid, tanh, softmax).
    -   **Impact:** Users likely need to import activation *modules* (like Dropout) directly from `ember_ml.nn.modules.activations`.

### 3.2. Modules/Functions Missing Explicit Mention in Main Docs Index (vs. Code)

-   **Top-Level Submodules:**
    -   **Code:** `ember_ml/__init__.py` exposes `benchmarks`, `data`, `training`, `visualization`, `wave` in its `__all__` list (lines 76-84).
    -   **Docs:** `docs/api/index.md` does not explicitly list these in its main structure, though they might be covered under "Other Modules" or linked pages.

-   **Backend Setup Functions:**
    -   **Code:** `ember_ml/__init__.py` exposes `set_backend` and `auto_select_backend` (lines 74-75).
    -   **Docs:** `docs/api/index.md` does not list these in the main structure. They might be covered in the `frontend_usage_guide.md`.

### 3.3. Structural / Naming Differences

-   **Ops (`linearalg`, `stats`):**
    -   **Docs:** `docs/api/index.md` presents `ops.linearalg` and `ops.stats` as distinct documented modules (lines 8-9).
    -   **Code:** `ember_ml/ops/__init__.py` does not explicitly set up dispatch for these in the same way as `MathOps`, `VectorOps`, etc. A comment (line 27) suggests they are "imported directly by users". `SolverOps` and `linearalg_ops` appear as potentially non-functional remnants in `__all__`.
    -   **Impact:** The documentation structure implies these are accessed via `ember_ml.ops.linearalg` or `ember_ml.ops.stats`, while the code suggests imports might be different (e.g., `from ember_ml import linearalg`). This needs clarification.

-   **Initializers:**
    -   **Docs:** `docs/api/index.md` lists `ember_ml.initializers` (line 23).
    -   **Code:** `ember_ml/__init__.py` does *not* list `initializers` in its `__all__`.
    -   **Impact:** Users need to import directly from `ember_ml.initializers`.

-   **NN Wiring:**
    -   **Docs:** `docs/api/index.md` lists `ember_ml.nn.modules.wiring` (Neuron Maps) (line 19).
    -   **Code:** `ember_ml/nn/__init__.py` imports `NeuronMap` subclasses directly from `ember_ml.nn.modules` (which re-exports from `wiring`) and includes them in `__all__`.
    -   **Impact:** Minor difference. Docs reflect the underlying module structure, while code flattens the import path slightly for these specific classes.

### 3.4. Minor `__all__` Inconsistencies in `ops/__init__.py`

-   `SolverOps` (line 231), `solver_ops` (line 241), `linearalg_ops` (line 245) are present but seem to lack corresponding interface imports or functional definitions within the file.
-   `gradient` (lines 247, 280) and `eigh` (lines 281, 282) are listed twice.
-   `'gaussian'` (line 330) appears, likely a typo or misplaced entry among FFT functions.

## 4. Recommendations

-   Update `docs/api/index.md` and related pages to accurately reflect how users should import modules/classes, especially for RNNs, features, activations, linear algebra, and statistics. Clarify if `linearalg` and `stats` are intended to be imported from the top level or via `ops`.
-   Consider adding RNNs, features, and activations to the `__all__` list in `nn/__init__.py` if direct import `from ember_ml.nn import ...` is the desired user experience, or update docs to explicitly show `from ember_ml.nn.modules...` imports.
-   Add missing top-level modules (`benchmarks`, `data`, etc.) to the main documentation index if they are considered part of the public API. Clarify the status of `initializers`.
-   Clean up the `__all__` list in `ops/__init__.py` to remove remnants, duplicates, and typos.
-   Ensure the documentation for `set_backend` and `auto_select_backend` is clearly visible, perhaps in the main index or a dedicated setup/usage guide.