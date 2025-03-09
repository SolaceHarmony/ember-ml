# NumPy Purification Plan for EmberHarmony

## Overview

This document outlines a comprehensive plan to purify the EmberHarmony codebase by removing direct NumPy usage from all files except for `emberharmony.backend.numpy`. The goal is to ensure that all tensor operations use the ops abstraction layer, making the code truly backend-agnostic.

## Current Status

Based on the analysis using `utils/detect_numpy_usage.py`, the current status is:

- **Total files analyzed**: 479
- **Files with NumPy**: 236 (49.27%)
- **Files with precision-reducing casts**: 100 (20.88%)
- **Files with tensor conversions**: 98 (20.46%)
- **Files with Python operators**: 302 (63.05%)

## Purification Principles

1. **Only emberharmony.backend.numpy should import NumPy directly**
2. **All tensor operations should use the ops abstraction layer**
3. **Python operators (+, -, *, /, etc.) should be replaced with ops functions**
4. **Precision-reducing casts (float(), int()) should be avoided**
5. **Tensor conversions between backends should be eliminated**
6. **Special cases where tools require NumPy should be tested first**

## Prioritization Strategy

Files will be prioritized for purification based on the following criteria:

1. **Core components**: Files in the core modules that are used by many other components
2. **High-impact components**: Files that are frequently used in the codebase
3. **Public API components**: Files that are part of the public API
4. **Examples and tests**: Files that demonstrate usage or test functionality

## Phase 1: Core Components

### emberharmony/core

The core module contains fundamental components that are used throughout the codebase. Purifying these files first will have the greatest impact.

- [x] emberharmony/core/base.py
- [x] emberharmony/core/ltc.py
- [x] emberharmony/core/spherical_ltc.py
- [x] emberharmony/core/geometric.py
- [ ] emberharmony/core/blocky.py
- [ ] emberharmony/core/hybrid.py
- [ ] emberharmony/core/stride_aware_cfc.py

### emberharmony/nn/wirings

The wirings module is critical for neural circuit policies and is used by many components.

- [ ] emberharmony/nn/wirings/wiring.py
- [ ] emberharmony/nn/wirings/ncp.py
- [ ] emberharmony/nn/wirings/ncp_wiring.py
- [ ] emberharmony/nn/wirings/auto_ncp.py
- [ ] emberharmony/nn/wirings/full_wiring.py
- [ ] emberharmony/nn/wirings/random_wiring.py

### emberharmony/nn/modules

The modules directory contains key neural network components.

- [ ] emberharmony/nn/modules/ncp.py
- [ ] emberharmony/nn/modules/auto_ncp.py

## Phase 2: High-Impact Components

### emberharmony/features

The features module is used for feature extraction and is a critical part of the pipeline.

- [ ] emberharmony/features/column_feature_extraction.py
- [ ] emberharmony/features/feature_engineer.py
- [ ] emberharmony/features/bigquery_feature_extraction.py
- [ ] emberharmony/features/generic_feature_extraction.py
- [ ] emberharmony/features/terabyte_feature_extractor_bigframes.py

### emberharmony/models

The models module contains high-level models that are used in applications.

- [ ] emberharmony/models/rbm.py
- [ ] emberharmony/models/optimized_rbm.py
- [ ] emberharmony/models/rbm_anomaly_detector.py
- [ ] emberharmony/models/rbm/rbm.py
- [ ] emberharmony/models/rbm/rbm_example.py
- [ ] emberharmony/models/rbm/rbm_backend_example.py
- [ ] emberharmony/models/rbm/rbm_anomaly_detection_example.py
- [ ] emberharmony/models/liquid/liquidtrainer.py
- [ ] emberharmony/models/liquid/liquid_anomaly_detector.py

### emberharmony/wave

The wave module contains wave-based neural network components.

- [ ] emberharmony/wave/binary/binary_wave_processor.py
- [ ] emberharmony/wave/binary/binary_wave_neuron.py
- [ ] emberharmony/wave/binary/wave_interference_processor.py
- [ ] emberharmony/wave/binary/binary_exact_processor.py
- [ ] emberharmony/wave/limb/limb_wave_processor.py
- [ ] emberharmony/wave/limb/wave_segment.py
- [ ] emberharmony/wave/limb/pwm_processor.py
- [ ] emberharmony/wave/limb/hpc_limb_core.py
- [ ] emberharmony/wave/harmonic/wave_generator.py
- [ ] emberharmony/wave/harmonic/training.py
- [ ] emberharmony/wave/harmonic/visualization.py
- [ ] emberharmony/wave/harmonic/embedding_utils.py
- [ ] emberharmony/wave/memory/multi_sphere.py
- [ ] emberharmony/wave/memory/sphere_overlap.py
- [ ] emberharmony/wave/memory/math_helpers.py
- [ ] emberharmony/wave/memory/metrics.py
- [ ] emberharmony/wave/memory/visualizer.py
- [ ] emberharmony/wave/utils/math_helpers.py
- [ ] emberharmony/wave/utils/wave_analysis.py
- [ ] emberharmony/wave/utils/wave_conversion.py
- [ ] emberharmony/wave/utils/wave_visualization.py
- [ ] emberharmony/wave/generator.py
- [ ] emberharmony/wave/harmonic.py

## Phase 3: Public API Components

### emberharmony/attention

The attention module contains attention mechanisms used in neural networks.

- [ ] emberharmony/attention/causal.py
- [ ] emberharmony/attention/multiscale_ltc.py
- [ ] emberharmony/attention/testfile.py
- [ ] emberharmony/attention/mechanisms/mechanism.py

### emberharmony/nn/specialized

The specialized module contains specialized neural network components.

- [ ] emberharmony/nn/specialized/base.py
- [ ] emberharmony/nn/specialized/specialized.py
- [ ] emberharmony/nn/specialized/attention.py

### emberharmony/nn/cfc

The CfC (Closed-form Continuous-time) module contains CfC neural network components.

- [ ] emberharmony/nn/cfc/stride_ware_cfc.py

### emberharmony/training

The training module contains components for training neural networks.

- [ ] emberharmony/training/hebbian/hebbian.py

## Phase 4: Examples and Tests

### examples

The examples directory contains example usage of the EmberHarmony framework.

- [ ] examples/backend_auto_selection_demo.py
- [ ] examples/bigquery_feature_extraction_demo.py
- [ ] examples/emberharmony_example.py
- [ ] examples/ncp_example.py
- [ ] examples/nn_example.py
- [ ] examples/otherneurons.py
- [ ] examples/purified_feature_extractor_demo.py

### tests

The tests directory contains tests for the EmberHarmony framework.

- [ ] tests/test_backend_auto_selection.py
- [ ] tests/test_backend.py
- [ ] tests/test_bigquery_feature_extraction.py
- [ ] tests/test_column_feature_extraction.py
- [ ] tests/test_compare_random_ops_purified.py
- [ ] tests/test_detect_numpy_usage.py
- [ ] tests/test_ncp_pytest.py
- [ ] tests/test_ncp.py
- [ ] tests/test_ops_device.py
- [ ] tests/test_ops_dtype.py
- [ ] tests/test_ops_math.py
- [ ] tests/test_ops_random.py
- [ ] tests/test_ops_tensor.py
- [ ] tests/test_terabyte_feature_extractor_purified_v2.py
- [ ] tests/test_terabyte_feature_extractor_purified.py

## Implementation Approach

For each file, the following steps will be taken:

1. **Analyze the file** using `utils/detect_numpy_usage.py` to identify NumPy usage, precision-reducing casts, tensor conversions, and Python operators
2. **Replace NumPy imports** with `from emberharmony import ops`
3. **Replace NumPy functions** with ops functions
4. **Replace Python operators** with ops functions
5. **Avoid precision-reducing casts** by using ops functions
6. **Eliminate tensor conversions** by staying within the ops abstraction layer
7. **Test the file** to ensure it works correctly with all backends
8. **Update documentation** to reflect the changes

## Special Cases

Some files may require NumPy for integration with external libraries. In these cases:

1. **Test if NumPy is actually required** by trying to use ops functions first
2. **Isolate NumPy usage** to specific functions that require it
3. **Document the reason** for using NumPy directly
4. **Add a comment** explaining why NumPy is used directly

## Testing Strategy

After purifying each file, the following tests should be run:

1. **Unit tests** for the specific component
2. **Integration tests** that use the component
3. **Backend compatibility tests** to ensure the component works with all backends
4. **Performance tests** to ensure the component performs well with all backends

## Documentation

For each purified file, the following documentation should be updated:

1. **API documentation** to reflect the changes
2. **Usage examples** to demonstrate the new usage
3. **Migration guides** for users who need to update their code

## Timeline

The purification process will be carried out in phases, with each phase focusing on a specific set of components. The estimated timeline is:

- **Phase 1**: 2 weeks
- **Phase 2**: 3 weeks
- **Phase 3**: 2 weeks
- **Phase 4**: 1 week

Total: 8 weeks

## Conclusion

This plan provides a comprehensive approach to purifying the EmberHarmony codebase by removing direct NumPy usage from all files except for `emberharmony.backend.numpy`. By following this plan, the codebase will become truly backend-agnostic, allowing it to run efficiently on different hardware without modification.