# Feature Extraction Module (`nn.features`)

The `ember_ml.nn.features` module provides components for feature extraction and transformation. It combines stateful components (classes like `PCA`, `Standardize`, `Normalize`) with stateless, backend-agnostic operations (like `one_hot`).

## Importing

```python
from ember_ml.nn import features
from ember_ml.nn import tensor # For creating example tensors
```

## Stateful Feature Components

These components maintain internal state (e.g., fitted PCA components, standardization means/stds) and are typically used in a fit/transform pattern. They are instantiated via factory functions or directly using their class names.

### PCA

Performs Principal Component Analysis for dimensionality reduction.

**Instantiation:**
```python
# Using the factory function (recommended)
pca_instance = features.pca()

# Direct instantiation
from ember_ml.nn.features import PCA
pca_instance = PCA(n_components=2)
```

**Usage:**
```python
# Fit PCA to data
data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
pca_instance.fit(data)

# Transform data
transformed = pca_instance.transform(data)

# Inverse transform
reconstructed = pca_instance.inverse_transform(transformed)
```
**Key Methods:** `fit`, `transform`, `fit_transform`, `inverse_transform`

### Standardize

Standardizes features by removing the mean and scaling to unit variance.

**Instantiation:**
```python
# Using the factory function (recommended)
std_scaler = features.standardize()

# Direct instantiation
from ember_ml.nn.features import Standardize
std_scaler = Standardize(with_mean=True, with_std=True)
```

**Usage:**
```python
# Fit the scaler
data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
std_scaler.fit(data)

# Transform data
standardized_data = std_scaler.transform(data)

# Inverse transform
original_data = std_scaler.inverse_transform(standardized_data)
```
**Key Methods:** `fit`, `transform`, `fit_transform`, `inverse_transform`

### Normalize

Normalizes features using various normalization techniques (e.g., L1, L2, max).

**Instantiation:**
```python
# Using the factory function (recommended)
normalizer = features.normalize()

# Direct instantiation
from ember_ml.nn.features import Normalize
normalizer = Normalize(norm='l2', axis=1) # Example: L2 norm along rows
```

**Usage:**
```python
# Fit the normalizer (often not needed for simple normalization)
data = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
# normalizer.fit(data) # Usually no-op

# Transform data
normalized_data = normalizer.transform(data)
```
**Key Methods:** `fit`, `transform`, `fit_transform`

## Stateless Feature Operations

These are backend-agnostic functions for common feature transformations.

### `features.one_hot(indices, depth, **kwargs)`

Convert integer indices to a one-hot encoded representation.

**Parameters:**
- `indices`: Tensor containing indices to convert.
- `depth`: The number of classes (determines the length of the one-hot vector).
- `**kwargs`: Backend-specific arguments.

**Returns:**
- One-hot encoded tensor (native backend tensor).

**Example:**
```python
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

## Specialized Feature Extractors (Classes)

Ember ML also includes specialized classes for common feature extraction pipelines, typically imported directly.

### `TerabyteFeatureExtractor`

Designed for extracting features from very large datasets, often involving chunking and out-of-core processing.

```python
from ember_ml.nn.features.terabyte_feature_extractor import TerabyteFeatureExtractor

extractor = TerabyteFeatureExtractor(
    window_size=100,
    stride=10,
    feature_functions=['mean', 'std', 'min', 'max']
)

# Extract features (example assumes time_series_data is loaded)
# features = extractor.extract(time_series_data)
```

### `TemporalStrideProcessor`

Processes temporal data with variable strides, useful for time series analysis.

```python
from ember_ml.nn.features.temporal_stride_processor import TemporalStrideProcessor

processor = TemporalStrideProcessor(
    stride_lengths=[1, 2, 4, 8],
    feature_functions=['mean', 'std', 'skew', 'kurtosis']
)

# Process temporal data (example assumes temporal_data is loaded)
# features = processor.process(temporal_data)
```

### `ColumnFeatureExtractor`

Extracts features from tabular data on a column-by-column basis, handling categorical, numerical, and text columns.

```python
from ember_ml.nn.features.column_feature_extractor import ColumnFeatureExtractor

extractor = ColumnFeatureExtractor(
    categorical_columns=['gender', 'country'],
    numerical_columns=['age', 'income', 'height'],
    text_columns=['description']
)

# Extract features from tabular data (example assumes tabular_data is loaded)
# features = extractor.extract(tabular_data)
```

## Backend Support

All feature extraction operations and components are designed to be backend-agnostic, leveraging the `ops` module internally where necessary. Stateful components like `PCA` manage their state independently of the backend, while stateless functions like `one_hot` rely on the dynamically aliased backend implementation.

## Notes

- For basic tensor creation and manipulation, use the `ember_ml.nn.tensor` module.
- For mathematical and statistical operations within custom feature functions, use the `ember_ml.ops` and `ember_ml.ops.stats` modules respectively.
- Refer to specific class documentation for detailed parameters and methods of `PCA`, `Standardize`, `Normalize`, `TerabyteFeatureExtractor`, `TemporalStrideProcessor`, and `ColumnFeatureExtractor`.