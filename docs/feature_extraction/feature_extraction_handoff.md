# Feature Extraction and Neural Network Pipeline Architecture

## Overview

This document outlines the architecture for a data processing pipeline that:

1. Uses BigFrames to extract features from terabyte-sized tables in BigQuery
2. Processes these features through Restricted Boltzmann Machines (RBMs)
3. Feeds the RBM output into a CfC-based liquid neural network with LSTM neurons for gating
4. Implements a motor neuron that outputs a value to trigger deeper data exploration

The pipeline leverages existing components from the emberharmony library, including AutoNCP for neural circuit policies.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │
│  BigQuery Data  │────▶│ Feature         │────▶│ RBM Feature     │────▶│ CfC Liquid      │
│  (Terabyte      │     │ Extraction      │     │ Learning        │     │ Neural Network  │
│   Tables)       │     │ (BigFrames)     │     │                 │     │                 │
│                 │     │                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                │
                                                                                │
                                                                                ▼
                                                                        ┌─────────────────┐
                                                                        │                 │
                                                                        │ Motor Neuron    │
                                                                        │ Output          │
                                                                        │                 │
                                                                        └─────────────────┘
```

## Component Details

### 1. BigQuery Data Preparation

- **Input**: Terabyte-sized tables in BigQuery
- **Process**: 
  - Connect to BigQuery using service account credentials
  - Query and extract data with appropriate filtering
  - Handle schema detection and type conversion
- **Output**: Structured data ready for feature extraction
- **Key Components**:
  - `prepare_bigquery_data.ipynb` notebook
  - BigFrames pandas API

### 2. Feature Extraction

- **Input**: Structured data from BigQuery
- **Process**:
  - Detect column types (numeric, categorical, datetime, boolean)
  - Apply appropriate transformations for each type
  - Create temporal features using stride-aware processing
  - Apply dimensionality reduction where appropriate
- **Output**: Feature vectors suitable for RBM training
- **Key Components**:
  - `BigQueryFeatureExtractor` class
  - `BigQueryTemporalStrideProcessor` class
  - `ColumnFeatureExtractor` class

### 3. RBM Feature Learning

- **Input**: Feature vectors from the extraction phase
- **Process**:
  - Train Restricted Boltzmann Machines to learn latent representations
  - Apply contrastive divergence for training
  - Extract learned features from the hidden layer
- **Output**: Latent feature representations
- **Key Components**:
  - `RestrictedBoltzmannMachine` class
  - `RBM` class (PyTorch implementation)

### 4. CfC Liquid Neural Network

- **Input**: Latent features from RBM
- **Process**:
  - Process through a Closed-form Continuous-time (CfC) neural network
  - Apply LSTM neurons for gating
  - Use AutoNCP for neural circuit policy wiring
- **Output**: Processed signals for the motor neuron
- **Key Components**:
  - `StrideAwareCfC` class
  - `AutoNCP` class for wiring configuration

### 5. Motor Neuron Output

- **Input**: Processed signals from the liquid neural network
- **Process**:
  - Aggregate signals through a final layer
  - Apply activation function to produce output value
  - Implement threshold for triggering deeper exploration
- **Output**: Value that triggers deeper exploration when threshold is exceeded
- **Key Components**:
  - Custom motor neuron implementation

## Data Flow

1. **Data Extraction**:
   - Connect to BigQuery using service account
   - Extract data with appropriate filtering
   - Split into train/validation/test sets

2. **Feature Processing**:
   - Apply type-specific transformations
   - Create temporal features with different stride lengths
   - Apply PCA for dimensionality reduction

3. **RBM Training**:
   - Initialize RBM with appropriate architecture
   - Train using contrastive divergence
   - Extract learned features

4. **Liquid Neural Network**:
   - Configure CfC network with AutoNCP wiring
   - Process RBM features through the network
   - Apply LSTM gating mechanisms

5. **Output Generation**:
   - Process liquid neural network output through motor neuron
   - Apply threshold for triggering deeper exploration
   - Return result

## Implementation Plan

### Phase 1: Data Preparation and Feature Extraction

1. Extend `prepare_bigquery_data.ipynb` to handle terabyte-sized tables
2. Implement efficient chunking and processing strategies
3. Optimize feature extraction for large-scale data
4. Implement and test temporal stride processing

### Phase 2: RBM Implementation

1. Configure RBM architecture for the extracted features
2. Implement efficient training procedure
3. Optimize for large-scale data processing
4. Implement feature extraction from trained RBM

### Phase 3: Liquid Neural Network

1. Configure CfC network with appropriate parameters
2. Implement LSTM gating mechanisms
3. Configure AutoNCP wiring for the network
4. Optimize for processing RBM features

### Phase 4: Motor Neuron and Integration

1. Implement motor neuron output layer
2. Configure threshold for triggering deeper exploration
3. Integrate all components into a cohesive pipeline
4. Implement monitoring and logging

### Phase 5: Testing and Optimization

1. Test pipeline with sample data
2. Optimize performance bottlenecks
3. Implement error handling and recovery
4. Document the entire pipeline

## Technical Considerations

### Scalability

- Use BigFrames for distributed processing of terabyte-sized tables
- Implement chunking strategies for RBM training
- Optimize memory usage throughout the pipeline

### Performance

- Use efficient implementations of RBM and CfC
- Leverage GPU acceleration where available
- Implement parallel processing where appropriate

### Monitoring

- Implement logging throughout the pipeline
- Track performance metrics for each component
- Monitor resource usage for large-scale processing

## Next Steps

1. Review and refine this architecture
2. Develop detailed implementation plan for each phase
3. Identify potential bottlenecks and mitigation strategies
4. Begin implementation of Phase 1