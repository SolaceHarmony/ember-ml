# Liquid Neural Network Demo Project

This repository contains code, documentation, and examples for working with Liquid Neural Networks (LNN) and related components.

## Overview

The project focuses on several key areas:

1. **Feature Extraction**: Tools for extracting features from BigQuery tables for use in neural networks
2. **Notebook Integration**: Fixes and tools for working with Jupyter notebooks
3. **Backend Purification**: Implementation of backend-agnostic tensor operations
4. **Architecture**: Design and implementation of liquid neural networks

## Documentation

All documentation has been organized into the following directories:

- **[docs/notebook](docs/notebook/)**: Documentation related to notebook issues and fixes
- **[docs/feature_extraction](docs/feature_extraction/)**: Documentation for feature extraction components
- **[docs/architecture](docs/architecture/)**: Documentation for system architecture and purification
- **[docs/testing](docs/testing/)**: Documentation for testing procedures

## Key Components

### Feature Extraction

The project includes tools for extracting features from BigQuery tables, including:

- `TerabyteFeatureExtractor`: Extracts features from BigQuery tables
- `TerabyteTemporalStrideProcessor`: Processes temporal data with variable strides

### Notebook Integration

Tools for working with Jupyter notebooks, including:

- Notebook simulation scripts
- Fixes for common notebook issues
- Documentation on best practices

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

- `fix_notebook.sh`: Shell script to apply notebook fixes
- `update_notebook.py`: Python script that updates notebooks with fixed code
- `run_notebook_simulation.py`: Script to run notebook cells in sequence
- `run_purification_tests_v2.py`: Script to test the purified implementation
- `run_bigquery_pipeline.py`: Script to run the BigQuery pipeline

## Running the Notebook Simulation

To understand how the notebook fixes work, you can run the notebook simulation:

```bash
python run_notebook_simulation.py
```

This will run all the cells in sequence and show the output of each cell.

## Documentation

For more information about the project, see the documentation in the `docs/` directory:

- [Notebook Documentation](docs/notebook/)
- [Feature Extraction Documentation](docs/feature_extraction/)
- [Architecture Documentation](docs/architecture/)
- [Testing Documentation](docs/testing/)
