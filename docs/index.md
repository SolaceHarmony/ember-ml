# Ember ML Documentation

Welcome to the Ember ML documentation. Ember ML is a library for efficient feature extraction and processing of terabyte-scale datasets, with a focus on performance, scalability, and ease of use.

## Documentation Sections

- [Architecture](architecture/index.md): System architecture and design principles
- [Feature Extraction](feature_extraction/index.md): Documentation for feature extraction components
- [Notebook](notebook/index.md): Documentation related to Jupyter notebooks and fixes
- [Testing](testing/index.md): Testing procedures and test plans
- [API Reference](api/index.md): Detailed API documentation
- [Tutorials](tutorials/index.md): Step-by-step guides for common tasks
- [Examples](examples/index.md): Code examples and use cases
- [Troubleshooting](troubleshooting/index.md): Solutions for common issues
- [Development](development/index.md): Guidelines for contributors

## Quick Start

### Installation

```bash
pip install ember-ml
```

### Basic Usage

```python
import ember_ml as eh
from ember_ml import ops

# Create a feature extractor
extractor = eh.features.TerabyteFeatureExtractor(
    project_id="your-project-id",
    location="US"
)

# Extract features
result = extractor.prepare_data(
    table_id="your-dataset.your-table",
    target_column="your-target-column"
)

# Unpack results
train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer = result

# Convert to Ember ML tensors for GPU acceleration
train_tensor = ops.convert_to_tensor(train_df[train_features].values)
```

For more detailed instructions, see the [Getting Started](tutorials/getting_started.md) guide.

## Key Features

- **Backend Abstraction**: Automatically selects the optimal computational backend (MLX, PyTorch, or NumPy)
- **Terabyte-Scale Processing**: Efficiently handles very large datasets with streaming and chunking
- **Memory Optimization**: Minimizes memory usage for large-scale operations
- **Feature Engineering**: Comprehensive tools for feature extraction and transformation
- **GPU Acceleration**: Leverages GPU resources when available for faster processing

## Architecture Overview

Ember ML is designed with a modular architecture that separates concerns and promotes reusability:

- **Backend System**: Provides a unified API across different computational backends
- **Data Processing**: Handles efficient loading and processing of large datasets
- **Feature Extraction**: Implements feature engineering and transformation
- **Model Integration**: Connects with machine learning models for end-to-end workflows

For more details, see the [Architecture Documentation](architecture/index.md).

## Use Cases

Ember ML is particularly well-suited for:

- **Large-Scale Data Processing**: Efficiently process terabyte-scale datasets
- **Feature Engineering**: Extract and transform features for machine learning
- **BigQuery Integration**: Seamlessly work with BigQuery data
- **GPU-Accelerated Processing**: Leverage GPU resources for faster computation

## Getting Help

If you encounter any issues or have questions:

1. Check the [Troubleshooting](troubleshooting/index.md) guide
2. Search for similar issues in the [GitHub repository](https://github.com/your-org/ember-ml/issues)
3. Ask a question in the [Discussion forum](https://github.com/your-org/ember-ml/discussions)

## Contributing

We welcome contributions to Ember ML! See the [Development Guide](development/index.md) for information on how to contribute.

## License

Ember ML is released under the [MIT License](https://opensource.org/licenses/MIT).