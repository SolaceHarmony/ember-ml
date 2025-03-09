# Liquid Neural Network Demo Project

This repository contains code, documentation, and examples for working with Liquid Neural Networks (LNN) and related components.

## Overview

The project focuses on several key areas:

1. **Feature Extraction**: Tools for extracting features from BigQuery tables for use in neural networks
2. **Backend Purification**: Implementation of backend-agnostic tensor operations
3. **Architecture**: Design and implementation of liquid neural networks

## Documentation

All documentation has been organized into the following directories:

- **[docs/feature_extraction](docs/feature_extraction/)**: Documentation for feature extraction components
- **[docs/architecture](docs/architecture/)**: Documentation for system architecture and purification
- **[docs/testing](docs/testing/)**: Documentation for testing procedures

## Key Components

### Feature Extraction

The project includes tools for extracting features from BigQuery tables, including:

- `TerabyteFeatureExtractor`: Extracts features from BigQuery tables
- `TerabyteTemporalStrideProcessor`: Processes temporal data with variable strides

### Backend Purification

The project implements backend-agnostic tensor operations that can use different computational backends:

- MLX (optimized for Apple Silicon)
- PyTorch
- NumPy

## Getting Started

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Set up your Google Cloud credentials
4. Run the example scripts in the `examples/` directory

## Files in this Repository

- `run_purification_tests_v2.py`: Script to test the purified implementation
- `run_bigquery_pipeline.py`: Script to run the BigQuery pipeline

## Documentation

For more information about the project, see the documentation in the `docs/` directory:

- [Feature Extraction Documentation](docs/feature_extraction/)
- [Architecture Documentation](docs/architecture/)
- [Testing Documentation](docs/testing/)
