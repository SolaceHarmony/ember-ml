# Feature Extraction Module (nn.features)

The `ember_ml.nn.features` module provides a comprehensive set of feature extraction and transformation operations for machine learning tasks. These operations are backend-agnostic and follow a consistent API across different backends.

## Importing

```python
from ember_ml.nn import features
```

## Core Classes

### PCA

`PCA` performs principal component analysis, a dimensionality reduction technique.

```python
from ember_ml.nn.features import PCA
from ember_ml.nn import tensor

# Create a PCA instance
pca = PCA(n_components=2)

# Fit PCA to data
data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
pca.fit(data)

# Transform data
transformed = pca.transform(data)

# Inverse transform
reconstructed = pca.inverse_transform(transformed)
```

### StandardizeInterface

Standardizes features by removing the mean and scaling to unit variance.

### NormalizeInterface

Normalizes features using various normalization techniques like L1, L2, or max normalization.

### TensorFeaturesInterface

Provides tensor-specific feature operations like one-hot encoding.

## Common Operations

| Function | Description |
|----------|-------------|
| `features.fit(X, **kwargs)` | Fit the PCA model to the data |
| `features.transform(X, **kwargs)` | Apply dimensionality reduction to X |
| `features.fit_transform(X, **kwargs)` | Fit the model and apply dimensionality reduction |
| `features.inverse_transform(X, **kwargs)` | Transform data back to its original space |
| `features.one_hot(indices, depth, **kwargs)` | Convert indices to one-hot encoding |
| `features.scatter(indices, updates, shape, **kwargs)` | Scatter updates into a tensor |

## Examples

### Principal Component Analysis

```python
from ember_ml.nn import features, tensor
import numpy as np

# Create some data
data = tensor.convert_to_tensor(np.random.randn(100, 10))

# Fit and transform with PCA
transformed = features.fit_transform(data, n_components=3)
print(f"Original shape: {data.shape}")  # (100, 10)
print(f"Transformed shape: {transformed.shape}")  # (100, 3)

# Reconstruct the data
reconstructed = features.inverse_transform(transformed)
print(f"Reconstructed shape: {reconstructed.shape}")  # (100, 10)
```

### One-Hot Encoding

```python
from ember_ml.nn import features, tensor

# Create indices
indices = tensor.convert_to_tensor([0, 2, 1, 0])

# Convert to one-hot encoding
one_hot_encoded = features.one_hot(indices, depth=3)
print(one_hot_encoded)
# [[1, 0, 0],
#  [0, 0, 1],
#  [0, 1, 0],
#  [1, 0, 0]]
```

## Feature Extraction Components

Ember ML includes several specialized feature extraction components that are available through the nn.features module:

### TerabyteFeatureExtractor

Designed for extracting features from very large datasets.

```python
from ember_ml.nn.features import TerabyteFeatureExtractor

extractor = TerabyteFeatureExtractor(
    window_size=100,
    stride=10,
    feature_functions=['mean', 'std', 'min', 'max']
)

# Extract features from time series data
features = extractor.extract(time_series_data)
```

### TemporalStrideProcessor

Processes temporal data with variable strides, useful for time series analysis.

```python
from ember_ml.nn.features import TemporalStrideProcessor

processor = TemporalStrideProcessor(
    stride_lengths=[1, 2, 4, 8],
    feature_functions=['mean', 'std', 'skew', 'kurtosis']
)

# Process temporal data
features = processor.process(temporal_data)
```

### ColumnFeatureExtractor

Extracts features from tabular data on a column-by-column basis.

```python
from ember_ml.nn.features import ColumnFeatureExtractor

extractor = ColumnFeatureExtractor(
    categorical_columns=['gender', 'country'],
    numerical_columns=['age', 'income', 'height'],
    text_columns=['description']
)

# Extract features from tabular data
features = extractor.extract(tabular_data)
```

## Backend Support

The feature extraction operations are backend-agnostic and work with any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.

```python
from ember_ml.nn import features
from ember_ml.backend import set_backend

# Use NumPy backend
set_backend('numpy')
pca_numpy = features.PCA(n_components=2)

# Use PyTorch backend
set_backend('torch')
pca_torch = features.PCA(n_components=2)

# Use MLX backend
set_backend('mlx')
pca_mlx = features.PCA(n_components=2)
```

## Implementation Details

The feature extraction module is implemented using a layered architecture:

1. **Interfaces**: Define the API for feature extraction operations
2. **Common Implementations**: Provide backend-agnostic implementations
3. **Backend-Specific Implementations**: Optimize for specific backends

This architecture allows Ember ML to provide consistent feature extraction capabilities across different backends while still leveraging the unique capabilities of each backend.

## Feature Engineering Pipeline

Ember ML's feature extraction components can be combined to create powerful feature engineering pipelines:

```python
from ember_ml.nn.features import ColumnFeatureExtractor, TemporalStrideProcessor, PCA

# Extract features from tabular data
column_extractor = ColumnFeatureExtractor(
    categorical_columns=['gender', 'country'],
    numerical_columns=['age', 'income']
)
tabular_features = column_extractor.extract(tabular_data)

# Process temporal columns
temporal_processor = TemporalStrideProcessor(
    stride_lengths=[1, 2, 4],
    feature_functions=['mean', 'std']
)
temporal_features = temporal_processor.process(temporal_data)

# Combine features
combined_features = concatenate([tabular_features, temporal_features], axis=1)

# Apply dimensionality reduction
pca = PCA(n_components=10)
final_features = pca.fit_transform(combined_features)
```

This pipeline extracts features from tabular data, processes temporal data with variable strides, combines the features, and applies dimensionality reduction to create a compact feature representation.

## Advanced Usage

### Custom Feature Functions

You can define custom feature functions for use with the feature extraction components:

```python
from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.ops import stats

def entropy(x, axis=None):
    """Compute the entropy of a probability distribution."""
    x = tensor.convert_to_tensor(x)
    x = ops.clip(x, 1e-10, 1.0)
    return -ops.stats.sum(x * ops.log(x), axis=axis)

# Use the custom feature function
from ember_ml.nn.features import TemporalStrideProcessor

processor = TemporalStrideProcessor(
    stride_lengths=[1, 2, 4],
    feature_functions=['mean', 'std', entropy]
)
```

### Feature Selection

You can combine feature extraction with feature selection techniques:

```python
from ember_ml.nn.features import PCA
from ember_ml import ops

# Extract features
features = extract_features(data)

# Compute feature importance
importance = compute_feature_importance(features, labels)

# Select top k features
k = 10
top_indices = ops.argsort(importance)[-k:]
selected_features = features[:, top_indices]

# Apply PCA to the selected features
pca = PCA(n_components=5)
final_features = pca.fit_transform(selected_features)
```

## Notes

- All feature extraction operations are backend-agnostic and work with any backend.
- The operations follow a consistent API across different backends.
- For tensor creation and manipulation, use the `ember_ml.nn.tensor` module.
- For mathematical operations, use the `ember_ml.ops` module.
- For statistical operations, use the `ember_ml.ops.stats` module.