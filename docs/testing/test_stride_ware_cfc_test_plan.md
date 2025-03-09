# Test Plan for Stride-Aware CfC and Neural Network Pipeline

This document outlines the testing strategy for the stride-aware CfC-based liquid neural network pipeline, including feature extraction, RBM, and motor neuron components.

## 1. Unit Testing

### 1.1 BigQuery Feature Extraction

#### Test Cases:

1. **Connection Setup**
   - Test successful connection to BigQuery
   - Test handling of invalid credentials
   - Test connection with different project settings

2. **Data Type Detection**
   - Test detection of numeric columns
   - Test detection of categorical columns
   - Test detection of datetime columns
   - Test detection of boolean columns
   - Test handling of mixed types

3. **Feature Engineering**
   - Test scaling of numeric features
   - Test one-hot encoding of categorical features
   - Test cyclical encoding of datetime features
   - Test handling of missing values

4. **Chunked Processing**
   - Test processing of small chunks
   - Test processing of large chunks
   - Test handling of chunk boundaries
   - Test memory usage during chunked processing

#### Example Test:

```python
def test_numeric_feature_scaling():
    """Test scaling of numeric features."""
    # Create sample data
    data = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10, 20, 30, 40, 50]
    })
    
    # Create feature extractor
    extractor = BigQueryFeatureExtractor()
    
    # Fit and transform
    result = extractor.prepare_data(data, target_column=None)
    
    # Check that scaling was applied
    assert extractor.scaler is not None
    
    # Check that scaled values are in expected range
    scaled_data = result[0]
    assert scaled_data['numeric1'].max() <= 1.0
    assert scaled_data['numeric1'].min() >= -1.0
    assert scaled_data['numeric2'].max() <= 1.0
    assert scaled_data['numeric2'].min() >= -1.0
```

### 1.2 Temporal Stride Processing

#### Test Cases:

1. **Window Creation**
   - Test creation of windows with different sizes
   - Test creation of windows with different strides
   - Test handling of edge cases (small data, large windows)

2. **PCA Processing**
   - Test PCA dimensionality reduction
   - Test explained variance calculation
   - Test feature importance calculation
   - Test handling of constant columns

3. **Multi-Stride Integration**
   - Test processing with multiple stride perspectives
   - Test combination of stride perspectives
   - Test memory usage with multiple strides

#### Example Test:

```python
def test_window_creation():
    """Test creation of windows with different strides."""
    # Create sample data
    data = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10]
    ])
    
    # Create processor
    processor = TemporalStrideProcessor(window_size=2, stride_perspectives=[1, 2])
    
    # Process data
    result = processor._create_strided_sequences(data, stride=1)
    
    # Check result shape
    assert result.shape == (4, 2, 2)  # 4 windows, window size 2, 2 features
    
    # Check window content
    assert np.array_equal(result[0], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(result[1], np.array([[3, 4], [5, 6]]))
    assert np.array_equal(result[2], np.array([[5, 6], [7, 8]]))
    assert np.array_equal(result[3], np.array([[7, 8], [9, 10]]))
    
    # Test with stride 2
    result = processor._create_strided_sequences(data, stride=2)
    
    # Check result shape
    assert result.shape == (2, 2, 2)  # 2 windows, window size 2, 2 features
    
    # Check window content
    assert np.array_equal(result[0], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(result[1], np.array([[5, 6], [7, 8]]))
```

### 1.3 RBM Implementation

#### Test Cases:

1. **Initialization**
   - Test initialization with different parameters
   - Test weight initialization
   - Test bias initialization

2. **Forward Pass**
   - Test computation of hidden probabilities
   - Test sampling of hidden states
   - Test computation of visible probabilities
   - Test sampling of visible states

3. **Training**
   - Test contrastive divergence with k=1
   - Test contrastive divergence with k>1
   - Test weight updates
   - Test bias updates
   - Test momentum updates

4. **Feature Extraction**
   - Test transformation of input data
   - Test reconstruction of input data
   - Test reconstruction error calculation

#### Example Test:

```python
def test_rbm_initialization():
    """Test initialization of RBM."""
    # Create RBM
    rbm = RestrictedBoltzmannMachine(n_visible=10, n_hidden=5)
    
    # Check dimensions
    assert rbm.weights.shape == (10, 5)
    assert rbm.visible_bias.shape == (10,)
    assert rbm.hidden_bias.shape == (5,)
    
    # Check initialization values
    assert np.all(rbm.visible_bias == 0)
    assert np.all(rbm.hidden_bias == 0)
    assert np.all(np.abs(rbm.weights) < 0.1)  # Small random values

def test_contrastive_divergence():
    """Test contrastive divergence training."""
    # Create RBM
    rbm = RestrictedBoltzmannMachine(n_visible=4, n_hidden=2)
    
    # Create sample data
    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]
    ])
    
    # Save initial weights
    initial_weights = rbm.weights.copy()
    
    # Perform contrastive divergence
    error = rbm.contrastive_divergence(data, k=1)
    
    # Check that weights were updated
    assert not np.array_equal(rbm.weights, initial_weights)
    
    # Check that error is a float
    assert isinstance(error, float)
```

### 1.4 CfC Neural Network

#### Test Cases:

1. **Cell Initialization**
   - Test initialization of StrideAwareCfCCell
   - Test initialization of StrideAwareWiredCfCCell
   - Test initialization with different parameters

2. **Wiring Configuration**
   - Test AutoNCP wiring creation
   - Test wiring with different sparsity levels
   - Test wiring constraints

3. **Forward Pass**
   - Test cell forward pass
   - Test layer forward pass
   - Test handling of time scaling

4. **Training**
   - Test training with different optimizers
   - Test training with different loss functions
   - Test gradient flow through the network

#### Example Test:

```python
def test_stride_aware_cfc_cell_initialization():
    """Test initialization of StrideAwareCfCCell."""
    # Create cell
    cell = StrideAwareCfCCell(units=10, stride_length=2, time_scale_factor=1.5)
    
    # Check properties
    assert cell.units == 10
    assert cell.stride_length == 2
    assert cell.time_scale_factor == 1.5
    
    # Check state size
    assert cell.state_size == [10, 10]

def test_auto_ncp_wiring():
    """Test AutoNCP wiring creation."""
    # Create wiring
    wiring = AutoNCP(units=20, output_size=10, sparsity_level=0.5)
    
    # Check properties
    assert wiring.units == 20
    assert wiring.output_dim == 10
    assert wiring.sparsity_level == 0.5
    
    # Build wiring
    input_mask, recurrent_mask, output_mask = wiring.build()
    
    # Check mask shapes
    assert input_mask.shape == (wiring.input_dim,)
    assert recurrent_mask.shape == (20, 20)
    assert output_mask.shape == (20,)
    
    # Check sparsity
    assert np.mean(recurrent_mask) <= 0.5  # Should be sparse
```

### 1.5 Motor Neuron

#### Test Cases:

1. **Initialization**
   - Test initialization with different thresholds
   - Test initialization with different activation functions

2. **Forward Pass**
   - Test computation of motor neuron output
   - Test threshold comparison
   - Test trigger signal generation

3. **Adaptive Threshold**
   - Test threshold adaptation
   - Test history tracking
   - Test adaptation rate

#### Example Test:

```python
def test_motor_neuron_initialization():
    """Test initialization of MotorNeuron."""
    # Create motor neuron
    motor_neuron = MotorNeuron(threshold=0.7, activation='sigmoid')
    
    # Check properties
    assert motor_neuron.threshold == 0.7
    assert motor_neuron.activation_name == 'sigmoid'
    
    # Build layer
    motor_neuron.build(input_shape=(None, 10))
    
    # Check weights
    assert motor_neuron.kernel.shape == (10, 1)
    assert motor_neuron.bias.shape == (1,)

def test_motor_neuron_forward_pass():
    """Test forward pass of MotorNeuron."""
    # Create motor neuron
    motor_neuron = MotorNeuron(threshold=0.5, activation='sigmoid')
    motor_neuron.build(input_shape=(None, 3))
    
    # Set weights manually for testing
    motor_neuron.kernel.assign([[0.1], [0.2], [0.3]])
    motor_neuron.bias.assign([0.0])
    
    # Create input
    inputs = tf.constant([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    
    # Forward pass in training mode
    output = motor_neuron(inputs, training=True)
    
    # Check output shape
    assert output.shape == (2, 1)
    
    # Forward pass in inference mode
    output, trigger = motor_neuron(inputs, training=False)
    
    # Check output and trigger shapes
    assert output.shape == (2, 1)
    assert trigger.shape == (2, 1)
    
    # Check trigger values (should be 0 or 1)
    assert np.all(np.logical_or(trigger.numpy() == 0, trigger.numpy() == 1))
```

## 2. Integration Testing

### 2.1 Feature Extraction to RBM

#### Test Cases:

1. **Data Flow**
   - Test flow of data from feature extraction to RBM
   - Test handling of different feature types
   - Test handling of missing values

2. **Dimensionality**
   - Test compatibility of feature dimensions with RBM
   - Test handling of high-dimensional features

3. **Performance**
   - Test memory usage during integration
   - Test processing time for different data sizes

#### Example Test:

```python
def test_feature_extraction_to_rbm_integration():
    """Test integration of feature extraction with RBM."""
    # Create sample data
    data = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'datetime': pd.date_range(start='2023-01-01', periods=5)
    })
    
    # Create feature extractor
    extractor = ColumnFeatureExtractor()
    
    # Extract features
    features_df = extractor.fit_transform(data)
    
    # Convert to numpy array
    features = features_df.values
    
    # Create RBM
    rbm = RestrictedBoltzmannMachine(n_visible=features.shape[1], n_hidden=10)
    
    # Train RBM
    rbm.train(features, epochs=5)
    
    # Extract RBM features
    rbm_features = rbm.transform(features)
    
    # Check output shape
    assert rbm_features.shape == (5, 10)
```

### 2.2 RBM to Liquid Neural Network

#### Test Cases:

1. **Data Flow**
   - Test flow of data from RBM to liquid neural network
   - Test handling of different feature dimensions

2. **Multi-Stride Integration**
   - Test integration with multiple stride perspectives
   - Test combination of RBM features with stride processing

3. **Training**
   - Test end-to-end training of RBM and liquid neural network
   - Test convergence of the integrated system

#### Example Test:

```python
def test_rbm_to_liquid_network_integration():
    """Test integration of RBM with liquid neural network."""
    # Create sample data
    data = np.random.rand(100, 20)
    
    # Create RBM
    rbm = RestrictedBoltzmannMachine(n_visible=20, n_hidden=10)
    
    # Train RBM
    rbm.train(data, epochs=5)
    
    # Extract RBM features
    rbm_features = rbm.transform(data)
    
    # Create liquid neural network
    wiring = AutoNCP(units=16, output_size=8, sparsity_level=0.5)
    cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=1)
    cfc_layer = StrideAwareCfC(cell=cell, return_sequences=False)
    
    # Create model
    inputs = Input(shape=(None, 10))
    x = cfc_layer(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    # Reshape RBM features for sequence input
    sequence_data = rbm_features.reshape(rbm_features.shape[0], 1, rbm_features.shape[1])
    
    # Create target data
    targets = np.random.rand(100, 1)
    
    # Train model
    model.fit(sequence_data, targets, epochs=2, batch_size=10)
    
    # Make predictions
    predictions = model.predict(sequence_data)
    
    # Check output shape
    assert predictions.shape == (100, 1)
```

### 2.3 Liquid Neural Network to Motor Neuron

#### Test Cases:

1. **Data Flow**
   - Test flow of data from liquid neural network to motor neuron
   - Test handling of different output dimensions

2. **Threshold Mechanism**
   - Test threshold comparison
   - Test trigger signal generation
   - Test adaptive threshold adjustment

3. **End-to-End Integration**
   - Test complete flow from liquid neural network to motor neuron
   - Test feedback from motor neuron to liquid neural network

#### Example Test:

```python
def test_liquid_network_to_motor_neuron_integration():
    """Test integration of liquid neural network with motor neuron."""
    # Create liquid neural network
    wiring = AutoNCP(units=16, output_size=8, sparsity_level=0.5)
    cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=1)
    cfc_layer = StrideAwareCfC(cell=cell, return_sequences=False)
    
    # Create model with motor neuron
    inputs = Input(shape=(None, 10))
    x = cfc_layer(inputs)
    motor_output = MotorNeuron(threshold=0.5)(x)
    
    model = Model(inputs=inputs, outputs=motor_output)
    model.compile(optimizer='adam', loss='mse')
    
    # Create sample data
    sequence_data = np.random.rand(100, 1, 10)
    targets = np.random.rand(100, 8)
    
    # Train model
    model.fit(sequence_data, targets, epochs=2, batch_size=10)
    
    # Make predictions
    predictions = model.predict(sequence_data)
    
    # In inference mode, we get both output and trigger
    if isinstance(predictions, list):
        output, trigger = predictions
        
        # Check output shapes
        assert output.shape[0] == 100
        assert trigger.shape[0] == 100
        
        # Check trigger values (should be 0 or 1)
        assert np.all(np.logical_or(trigger == 0, trigger == 1))
```

## 3. End-to-End Testing

### 3.1 Complete Pipeline

#### Test Cases:

1. **Data Flow**
   - Test flow of data through the complete pipeline
   - Test handling of different data types and sizes

2. **Performance**
   - Test memory usage during end-to-end processing
   - Test processing time for different data sizes
   - Test scalability with increasing data size

3. **Accuracy**
   - Test accuracy of the complete pipeline
   - Test stability of results with different inputs

#### Example Test:

```python
def test_complete_pipeline():
    """Test the complete pipeline from feature extraction to motor neuron."""
    # Create sample data
    data = pd.DataFrame({
        'numeric1': np.random.rand(100),
        'numeric2': np.random.rand(100),
        'category': np.random.choice(['A', 'B', 'C'], size=100),
        'datetime': pd.date_range(start='2023-01-01', periods=100)
    })
    
    # Create pipeline components
    feature_extractor = ColumnFeatureExtractor()
    
    rbm = RestrictedBoltzmannMachine(
        n_visible=20,  # Estimated feature dimension
        n_hidden=10
    )
    
    wiring = AutoNCP(units=16, output_size=8, sparsity_level=0.5)
    cell = StrideAwareWiredCfCCell(wiring=wiring, stride_length=1)
    cfc_layer = StrideAwareCfC(cell=cell, return_sequences=False)
    
    # Create model with motor neuron
    inputs = Input(shape=(None, 10))
    x = cfc_layer(inputs)
    motor_output = MotorNeuron(threshold=0.5)(x)
    
    liquid_network = Model(inputs=inputs, outputs=motor_output)
    liquid_network.compile(optimizer='adam', loss='mse')
    
    # Create pipeline
    pipeline = LiquidNeuralPipeline(
        feature_extractor=feature_extractor,
        rbm=rbm,
        liquid_network=liquid_network
    )
    
    # Process data through pipeline
    try:
        # Extract features
        features_df = feature_extractor.fit_transform(data)
        
        # Train RBM
        rbm.train(features_df.values, epochs=5)
        
        # Extract RBM features
        rbm_features = rbm.transform(features_df.values)
        
        # Reshape for sequence input
        sequence_data = rbm_features.reshape(rbm_features.shape[0], 1, rbm_features.shape[1])
        
        # Create target data
        targets = np.random.rand(100, 8)
        
        # Train liquid network
        liquid_network.fit(sequence_data, targets, epochs=2, batch_size=10)
        
        # Make predictions
        predictions = liquid_network.predict(sequence_data)
        
        # Check that pipeline runs without errors
        assert True
    except Exception as e:
        assert False, f"Pipeline failed with error: {e}"
```

### 3.2 Real-World Data Testing

#### Test Cases:

1. **BigQuery Integration**
   - Test with real BigQuery tables
   - Test with different table sizes and schemas
   - Test with different query patterns

2. **Terabyte-Scale Processing**
   - Test with large datasets
   - Test chunked processing
   - Test memory usage and performance

3. **Long-Running Tests**
   - Test stability over long periods
   - Test with continuous data streams
   - Test recovery from failures

#### Example Test:

```python
def test_bigquery_integration():
    """Test integration with real BigQuery tables."""
    # Skip if no credentials available
    if not os.path.exists('credentials.json'):
        pytest.skip("No BigQuery credentials available")
    
    # Set up BigQuery connection
    project_id = 'test-project'
    table_id = 'test-dataset.test-table'
    
    # Create feature extractor
    extractor = BigQueryFeatureExtractor(project_id=project_id)
    
    # Extract features
    try:
        result = extractor.prepare_data(
            table_id=table_id,
            limit=1000  # Limit for testing
        )
        
        # Check that extraction succeeds
        assert result is not None
        
        # Unpack results
        train_bf_df, val_bf_df, test_bf_df, train_features, val_features, test_features, scaler, imputer = result
        
        # Check that we have data
        assert len(train_bf_df) > 0
        assert len(val_bf_df) > 0
        assert len(test_bf_df) > 0
        
        # Check that we have features
        assert len(train_features) > 0
        
        # Check that scaler and imputer are created
        assert scaler is not None
        assert imputer is not None
    except Exception as e:
        assert False, f"BigQuery integration failed with error: {e}"
```

## 4. Performance Testing

### 4.1 Memory Usage

#### Test Cases:

1. **Memory Profiling**
   - Test memory usage during feature extraction
   - Test memory usage during RBM training
   - Test memory usage during liquid neural network processing
   - Test memory usage during end-to-end pipeline execution

2. **Memory Optimization**
   - Test effect of chunked processing on memory usage
   - Test effect of garbage collection on memory usage
   - Test memory usage with different batch sizes

#### Example Test:

```python
def test_memory_usage():
    """Test memory usage during pipeline execution."""
    import tracemalloc
    import gc
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create sample data
    data = pd.DataFrame({
        'numeric1': np.random.rand(10000),
        'numeric2': np.random.rand(10000),
        'category': np.random.choice(['A', 'B', 'C'], size=10000),
        'datetime': pd.date_range(start='2023-01-01', periods=10000)
    })
    
    # Create feature extractor
    extractor = ColumnFeatureExtractor()
    
    # Extract features
    gc.collect()  # Force garbage collection before measurement
    snapshot1 = tracemalloc.take_snapshot()
    
    features_df = extractor.fit_transform(data)
    
    snapshot2 = tracemalloc.take_snapshot()
    
    # Calculate memory usage
    memory_usage = sum(stat.size for stat in snapshot2.compare_to(snapshot1, 'lineno'))
    print(f"Feature extraction memory usage: {memory_usage / 1024 / 1024:.2f} MB")
    
    # Check that memory usage is reasonable
    assert memory_usage < 1024 * 1024 * 1000  # Less than 1000 MB
    
    # Stop memory tracking
    tracemalloc.stop()
```

### 4.2 Processing Time

#### Test Cases:

1. **Time Profiling**
   - Test processing time for feature extraction
   - Test processing time for RBM training
   - Test processing time for liquid neural network processing
   - Test processing time for end-to-end pipeline execution

2. **Scalability**
   - Test processing time with increasing data size
   - Test processing time with increasing feature dimension
   - Test processing time with increasing model complexity

#### Example Test:

```python
def test_processing_time():
    """Test processing time during pipeline execution."""
    import time
    
    # Create sample data
    data_sizes = [1000, 5000, 10000]
    processing_times = []
    
    for size in data_sizes:
        data = pd.DataFrame({
            'numeric1': np.random.rand(size),
            'numeric2': np.random.rand(size),
            'category': np.random.choice(['A', 'B', 'C'], size=size),
            'datetime': pd.date_range(start='2023-01-01', periods=size)
        })
        
        # Create feature extractor
        extractor = ColumnFeatureExtractor()
        
        # Measure processing time
        start_time = time.time()
        features_df = extractor.fit_transform(data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        print(f"Processing time for {size} rows: {processing_time:.2f} seconds")
    
    # Check that processing time scales reasonably
    # Should be roughly linear or better
    ratio1 = processing_times[1] / processing_times[0]
    ratio2 = processing_times[2] / processing_times[1]
    
    expected_ratio1 = data_sizes[1] / data_sizes[0]
    expected_ratio2 = data_sizes[2] / data_sizes[1]
    
    # Allow for some overhead, so actual ratio can be up to 50% higher than expected
    assert ratio1 <= expected_ratio1 * 1.5
    assert ratio2 <= expected_ratio2 * 1.5
```

## 5. Test Automation

### 5.1 Continuous Integration

- Set up CI pipeline for automated testing
- Configure test environments with required dependencies
- Set up test reporting and monitoring

### 5.2 Test Coverage

- Track test coverage for all components
- Identify areas with insufficient coverage
- Add tests to improve coverage

### 5.3 Regression Testing

- Set up regression test suite
- Automate regression testing on code changes
- Monitor for performance regressions

## 6. Test Environment

### 6.1 Development Environment

- Local development environment with small test datasets
- Mock BigQuery for local testing
- GPU acceleration for neural network testing

### 6.2 Staging Environment

- Cloud-based staging environment
- Access to test BigQuery tables
- Scaled-down version of production environment

### 6.3 Production Environment

- Full-scale production environment
- Access to production BigQuery tables
- Monitoring and alerting for test failures

## 7. Test Schedule

### 7.1 Unit Tests

- Run on every code change
- Run as part of CI pipeline
- Run before merging code

### 7.2 Integration Tests

- Run daily on staging environment
- Run before major releases
- Run after significant code changes

### 7.3 End-to-End Tests

- Run weekly on staging environment
- Run before production deployment
- Run after major system changes

### 7.4 Performance Tests

- Run weekly on staging environment
- Run before production deployment
- Run after performance-related changes

## 8. Test Deliverables

- Test plan document
- Test case specifications
- Test scripts and code
- Test results and reports
- Performance benchmarks
- Test coverage reports

## 9. Risks and Mitigations

### 9.1 Data Availability

- **Risk**: Test data may not be representative of production data
- **Mitigation**: Use anonymized production data for testing

### 9.2 Resource Constraints

- **Risk**: Testing large-scale data may require significant resources
- **Mitigation**: Use chunked processing and optimize resource usage

### 9.3 Test Environment Stability

- **Risk**: Test environment may not be stable
- **Mitigation**: Set up monitoring and alerting for test environment

## 10. Conclusion

This test plan provides a comprehensive approach to testing the stride-aware CfC-based liquid neural network pipeline. By following this plan, we can ensure that all components work correctly individually and together, and that the system performs well with large-scale data.