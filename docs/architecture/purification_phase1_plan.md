# emberharmony Purification

This project implements the first phase of the emberharmony purification plan, focusing on replacing direct NumPy usage with emberharmony's backend abstraction system.

## Overview

The purification initiative aims to improve the emberharmony codebase by:

1. Replacing direct NumPy usage with emberharmony's backend abstraction system
2. Improving BigQuery data handling for terabyte-scale datasets
3. Organizing notebook simulation tools in a dedicated directory
4. Centralizing documentation in a structured docs folder

This implementation focuses on the first goal: replacing direct NumPy usage with emberharmony's backend abstraction system.

## Implementation

The implementation consists of the following components:

1. **Backend Utilities**: A new module `emberharmony.utils.backend_utils` that provides utility functions for working with emberharmony's backend system.

2. **Purified TerabyteFeatureExtractor**: A new version of the TerabyteFeatureExtractor class that uses the backend utilities instead of direct NumPy calls.

3. **Tests**: A comprehensive test suite to verify that the purified version works correctly with different backends.

## Key Features

### Backend Abstraction

The purified version of TerabyteFeatureExtractor uses emberharmony's backend abstraction system, which automatically selects the optimal computational backend (MLX, PyTorch, or NumPy) based on availability. This allows the code to run efficiently on different hardware without modification.

```python
# Before (direct NumPy usage)
import numpy as np
df['__split_rand'] = np.random.rand(len(df))
df[f'{col}_sin_hour'] = np.sin(2 * np.pi * df[col].dt.hour / 23.0)

# After (backend-agnostic implementation)
from emberharmony.utils import backend_utils
random_values = backend_utils.random_uniform(len(df))
random_values_np = backend_utils.tensor_to_numpy_safe(random_values)
df['__split_rand'] = random_values_np

hours_sin, hours_cos = backend_utils.sin_cos_transform(df[col].dt.hour / 23.0)
df[f'{col}_sin_hour'] = backend_utils.tensor_to_numpy_safe(hours_sin)
```

### Backend Selection

The purified version allows you to specify a preferred backend when creating a TerabyteFeatureExtractor instance:

```python
# Create extractor with MLX backend (if available)
extractor = TerabyteFeatureExtractor(
    project_id="your-project-id",
    location="US",
    preferred_backend="mlx"
)

# Create extractor with PyTorch backend (if available)
extractor = TerabyteFeatureExtractor(
    project_id="your-project-id",
    location="US",
    preferred_backend="torch"
)

# Create extractor with NumPy backend
extractor = TerabyteFeatureExtractor(
    project_id="your-project-id",
    location="US",
    preferred_backend="numpy"
)
```

### Backend Utilities

The `emberharmony.utils.backend_utils` module provides utility functions for working with emberharmony's backend system:

- `get_current_backend()`: Get the current backend name
- `set_preferred_backend(backend_name)`: Set the preferred backend
- `initialize_random_seed(seed)`: Initialize random seed for reproducibility
- `convert_to_tensor_safe(data)`: Safely convert data to a tensor
- `tensor_to_numpy_safe(tensor)`: Safely convert a tensor to a NumPy array
- `random_uniform(shape, low, high)`: Generate uniform random values
- `sin_cos_transform(values, period)`: Apply sine and cosine transformations
- `vstack_safe(arrays)`: Safely stack arrays vertically
- `get_backend_info()`: Get information about the current backend
- `print_backend_info()`: Print information about the current backend

## Usage

### Basic Usage

```python
from emberharmony.features.terabyte_feature_extractor_purified import TerabyteFeatureExtractor

# Create extractor with preferred backend
extractor = TerabyteFeatureExtractor(
    project_id="your-project-id",
    location="US",
    preferred_backend="mlx"  # Try to use MLX if available
)

# Set up BigQuery connection
extractor.setup_bigquery_connection()

# Prepare data
result = extractor.prepare_data(
    table_id="your-dataset.your-table",
    target_column="your-target-column",
    force_categorical_columns=["category1", "category2"],
    limit=1000000  # For testing
)

# Unpack results
train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer = result

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Features: {train_features}")
```

### Running Tests

To run the tests for the purified version:

```bash
# Run all tests
./run_purification_tests.py

# Run a specific test
python -m unittest tests.test_terabyte_feature_extractor_purified.TestTerabyteFeatureExtractorPurified.test_backend_selection
```

## Benefits

The purified version of TerabyteFeatureExtractor provides several benefits:

1. **Performance**: Automatic GPU acceleration when available
2. **Flexibility**: Code that works across different hardware platforms
3. **Maintainability**: Consistent API regardless of backend
4. **Future-proofing**: Easy adoption of new backends as they become available

## Next Steps

This implementation is the first step in the larger emberharmony purification plan. Future steps include:

1. **BigQuery Streaming**: Implement true streaming processing for terabyte-scale datasets
2. **Notebook Tools Organization**: Organize notebook simulation tools in a dedicated directory
3. **Documentation Reorganization**: Centralize documentation in a structured docs folder

For more details, see the architectural documentation in the `docs/architecture/` directory.