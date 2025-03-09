# Feature Extraction Documentation

This directory contains documentation related to feature extraction components and processes used in the Liquid Neural Network project.

## Available Documentation

- [**Feature Extraction Handoff**](feature_extraction_handoff.md): Comprehensive handoff document for the feature extraction component
- [**Feature Extraction Implementation Plan**](feature_extraction_implementation_plan.md): Plan for implementing the feature extraction component
- [**Feature Extraction Summary**](feature_extraction_summary.md): Summary of the feature extraction process
- [**Image Feature Extractor Design**](image_feature_extractor_design.md): Design document for the image feature extractor
- [**Image Feature Extractor Implementation Plan**](image_feature_extractor_implementation_plan.md): Plan for implementing the image feature extractor
- [**Image Feature Extractor Test Plan**](image_feature_extractor_test_plan.md): Test plan for the image feature extractor

## Feature Extraction Components

The feature extraction system consists of several key components:

1. **TerabyteFeatureExtractor**: Extracts features from BigQuery tables, supporting both batch and streaming data processing modes.

2. **Temporal Stride Processing**: Implements a novel temporal stride mechanism for time-series data, allowing for variable-length temporal windows to capture different scales of patterns.

3. **Feature Transformation Pipeline**: Supports numerical feature normalization, categorical feature encoding, dimensionality reduction, and feature selection.

4. **Image Feature Extractor**: Extracts features from images for use in neural networks.

## BigQuery Integration

The feature extraction pipeline is tightly integrated with BigQuery for processing large-scale datasets:

- Uses BigFrames for efficient data manipulation within the BigQuery environment
- Supports both batch and streaming data processing modes
- Optimized for performance with large datasets through vectorized operations

## Usage Examples

For examples of how to use the feature extraction components, see the `examples/` directory in the root of the repository:

- `examples/bigquery_feature_extraction_demo.py`: Demonstrates how to extract features from BigQuery tables
- `examples/purified_feature_extractor_demo.py`: Demonstrates how to use the purified version of the feature extractor