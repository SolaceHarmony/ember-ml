# Backend Purification Implementation Guide

## Overview

This document provides a detailed implementation guide for replacing direct NumPy usage with emberharmony's backend abstraction system in the `terabyte_feature_extractor.py` file. This is part of the larger emberharmony purification plan.

## Background

emberharmony's backend system automatically selects the optimal computational backend (MLX, PyTorch, or NumPy) based on availability. This allows code to run efficiently on different hardware without modification. By using this system instead of direct NumPy calls, we can achieve:

1. Automatic GPU acceleration when available
2. Consistent API across different backends
3. Better performance on Apple Silicon via MLX
4. Simplified code maintenance

## Current State Analysis

The `terabyte_feature_extractor.py` file contains numerous direct NumPy calls:

| Category | NumPy Functions | Occurrences |
|----------|----------------|-------------|
| Array Creation | `np.array()`, `np.vstack()` | 12 |
| Random | `np.random.seed()`, `np.random.rand()` | 4 |
| Math | `np.sin()`, `np.cos()`, `np.pi` | 8 |
| Statistics | `np.mean()`, `np.min()`, `np.max()` | 6 |
| Array Manipulation | `np.abs()` | 2 |

## Implementation Strategy

### 1. Import Changes

Replace NumPy imports with emberharmony imports:

```python
# Before
import numpy as np

# After
from emberharmony import ops
from emberharmony.backend import get_backend, set_random_seed
```

### 2. Function Mapping

Use this mapping to replace NumPy functions with emberharmony equivalents:

| NumPy | emberharmony.ops | Notes |
|-------|-----------------|-------|
| `np.array()` | `ops.convert_to_tensor()` | Handles conversion from various types |
| `np.vstack()` | `ops.vstack()` | Vertical stack of arrays |
| `np.random.seed()` | `set_random_seed()` | Sets seed for all backends |
| `np.random.rand()` | `ops.random.uniform()` | Generate uniform random values |
| `np.sin()` | `ops.sin()` | Sine function |
| `np.cos()` | `ops.cos()` | Cosine function |
| `np.pi` | `ops.pi` | Pi constant |
| `np.mean()` | `ops.mean()` | Mean of array |
| `np.min()` | `ops.min()` | Minimum value |
| `np.max()` | `ops.max()` | Maximum value |
| `np.abs()` | `ops.abs()` | Absolute value |

### 3. Code Transformation Examples

#### Example 1: Random Number Generation

```python
# Before
np.random.seed(42)
df['__split_rand'] = np.random.rand(len(df))

# After
set_random_seed(42)
df['__split_rand'] = ops.random.uniform(size=len(df)).numpy()
```

#### Example 2: Trigonometric Functions

```python
# Before
df[f'{col}_sin_hour'] = np.sin(2 * np.pi * df[col].dt.hour / 23.0)
df[f'{col}_cos_hour'] = np.cos(2 * np.pi * df[col].dt.hour / 23.0)

# After
hours_tensor = ops.convert_to_tensor(df[col].dt.hour / 23.0)
df[f'{col}_sin_hour'] = ops.sin(2 * ops.pi * hours_tensor).numpy()
df[f'{col}_cos_hour'] = ops.cos(2 * ops.pi * hours_tensor).numpy()
```

#### Example 3: Array Operations

```python
# Before
batch_data = np.vstack([self.state_buffer, batch_data])
windows_array = np.array(windows)

# After
buffer_tensor = ops.convert_to_tensor(self.state_buffer)
batch_tensor = ops.convert_to_tensor(batch_data)
batch_data = ops.vstack([buffer_tensor, batch_tensor]).numpy()
windows_array = ops.convert_to_tensor(windows)
```

#### Example 4: Feature Importance Calculation

```python
# Before
return np.abs(self.pca_models[stride].components_).sum(axis=0)

# After
components = ops.convert_to_tensor(self.pca_models[stride].components_)
return ops.abs(components).sum(axis=0).numpy()
```

### 4. Pandas Integration

When working with pandas DataFrames, we need to convert between numpy arrays and emberharmony tensors:

```python
# Convert from DataFrame to tensor
tensor_data = ops.convert_to_tensor(df[features].values)

# Process with emberharmony ops
processed_data = ops.some_operation(tensor_data)

# Convert back to numpy for pandas
df[features] = processed_data.numpy()
```

### 5. Testing Strategy

For each converted function, implement tests that verify:

1. **Functional equivalence**: Results match the original NumPy implementation
2. **Backend switching**: Code works with different backends
3. **Performance**: Measure speed improvements with GPU backends

Example test:

```python
def test_trigonometric_conversion():
    # Test data
    hours = np.arange(24)
    df = pd.DataFrame({'hour': hours})
    
    # NumPy implementation
    df['sin_numpy'] = np.sin(2 * np.pi * df['hour'] / 23.0)
    
    # emberharmony implementation
    hours_tensor = ops.convert_to_tensor(df['hour'] / 23.0)
    df['sin_emberharmony'] = ops.sin(2 * ops.pi * hours_tensor).numpy()
    
    # Assert results are close
    np.testing.assert_allclose(df['sin_numpy'], df['sin_emberharmony'], rtol=1e-5)
```

## Implementation Plan

### Phase 1: Setup and Utilities (1 week)

1. Create utility functions for common operations:
   - `convert_to_tensor_safe`: Safely convert various inputs to tensors
   - `tensor_to_numpy_safe`: Safely convert tensors back to numpy arrays
   - `random_generator`: Unified random number generation

2. Implement backend detection and configuration:
   - Add backend detection to TerabyteFeatureExtractor initialization
   - Create configuration options for preferred backend

### Phase 2: Core Function Conversion (2 weeks)

1. Replace random number generation:
   - Update all `np.random` calls in `_split_data`
   - Test with different random seeds

2. Convert mathematical operations:
   - Replace trigonometric functions in `_create_datetime_features`
   - Update statistical operations in feature processing

3. Transform array operations:
   - Convert array creation and manipulation in `process_large_dataset`
   - Update PCA-related operations

### Phase 3: Integration and Testing (1 week)

1. Implement comprehensive tests:
   - Unit tests for each converted function
   - Integration tests for full processing pipeline
   - Performance benchmarks across backends

2. Create documentation:
   - Update docstrings with backend information
   - Add examples of backend-specific optimizations

## Code Examples for Key Functions

### TerabyteFeatureExtractor Initialization

```python
def __init__(
    self,
    project_id: Optional[str] = None,
    location: str = "US",
    chunk_size: int = 100000,
    max_memory_gb: float = 16.0,
    verbose: bool = True,
    preferred_backend: Optional[str] = None
):
    """
    Initialize the terabyte-scale feature extractor.
    
    Args:
        project_id: GCP project ID (optional if using in BigQuery Studio)
        location: BigQuery location (default: "US")
        chunk_size: Number of rows to process per chunk
        max_memory_gb: Maximum memory usage in GB
        verbose: Whether to print progress information
        preferred_backend: Preferred computation backend ('mlx', 'torch', 'numpy')
    """
    if not BIGFRAMES_AVAILABLE:
        raise ImportError("BigFrames is not available. Please install it to use TerabyteFeatureExtractor.")
    
    # Initialize backend
    from emberharmony.backend import get_backend, set_backend
    if preferred_backend:
        set_backend(preferred_backend)
    self.backend = get_backend()
    logger.info(f"Using {self.backend} backend for computation")
    
    # Set random seed for reproducibility
    from emberharmony.backend import set_random_seed
    set_random_seed(42)
    
    # Rest of initialization...
```

### Datetime Feature Creation

```python
def _create_datetime_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Create cyclical features from datetime column using emberharmony ops.
    
    Args:
        df: Input DataFrame
        col: Datetime column name
        
    Returns:
        DataFrame with added cyclical features
    """
    if col not in df.columns:
        return df
    
    # Ensure column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception as e:
            logger.warning(f"Could not convert {col} to datetime: {e}")
            return df
    
    from emberharmony import ops
    
    # Convert datetime components to tensors
    hours = ops.convert_to_tensor(df[col].dt.hour / 23.0)
    days_of_week = ops.convert_to_tensor(df[col].dt.dayofweek / 6.0)
    days_of_month = ops.convert_to_tensor((df[col].dt.day - 1) / 30.0)
    months = ops.convert_to_tensor((df[col].dt.month - 1) / 11.0)
    
    # Create cyclical features using sine and cosine transformations
    # Hour of day (0-23)
    df[f'{col}_sin_hour'] = ops.sin(2 * ops.pi * hours).numpy()
    df[f'{col}_cos_hour'] = ops.cos(2 * ops.pi * hours).numpy()
    
    # Day of week (0-6)
    df[f'{col}_sin_dayofweek'] = ops.sin(2 * ops.pi * days_of_week).numpy()
    df[f'{col}_cos_dayofweek'] = ops.cos(2 * ops.pi * days_of_week).numpy()
    
    # Day of month (1-31)
    df[f'{col}_sin_day'] = ops.sin(2 * ops.pi * days_of_month).numpy()
    df[f'{col}_cos_day'] = ops.cos(2 * ops.pi * days_of_month).numpy()
    
    # Month (1-12)
    df[f'{col}_sin_month'] = ops.sin(2 * ops.pi * months).numpy()
    df[f'{col}_cos_month'] = ops.cos(2 * ops.pi * months).numpy()
    
    logger.info(f"Created cyclical features for datetime column '{col}' using {self.backend} backend")
    return df
```

## Conclusion

This implementation guide provides a detailed roadmap for replacing NumPy with emberharmony's backend system in the `terabyte_feature_extractor.py` file. By following this guide, we can achieve better performance, more efficient resource utilization, and improved code maintainability.

The key benefits of this approach include:

1. **Performance**: Automatic GPU acceleration when available
2. **Flexibility**: Code that works across different hardware platforms
3. **Maintainability**: Consistent API regardless of backend
4. **Future-proofing**: Easy adoption of new backends as they become available

This implementation is the first step in the larger emberharmony purification plan, setting the foundation for a more robust and efficient codebase.