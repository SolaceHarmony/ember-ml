# ImageFeatureExtractor Test Plan

## 1. Testing Objectives

The testing strategy for the `ImageFeatureExtractor` aims to ensure:

1. **Correctness**: The extractor produces accurate and consistent features
2. **Robustness**: The extractor handles various input formats and edge cases
3. **Backend Compatibility**: The extractor works with all supported backends
4. **Integration**: The extractor integrates properly with the existing system
5. **Performance**: The extractor performs efficiently with various dataset sizes

## 2. Test Categories

### 2.1 Unit Tests

Unit tests will verify the correctness of individual components of the `ImageFeatureExtractor`.

#### 2.1.1 Initialization Tests

- Test initialization with default parameters
- Test initialization with custom parameters
- Test initialization with invalid parameters
- Test initialization with different strategies

```python
def test_initialization_default():
    """Test initialization with default parameters."""
    extractor = ImageFeatureExtractor()
    assert extractor.strategy == 'basic'
    assert extractor.include_color is True
    assert extractor.include_texture is True
    assert extractor.include_edges is True
    assert extractor.cnn_model is None
    assert extractor.target_size is None

def test_initialization_custom():
    """Test initialization with custom parameters."""
    extractor = ImageFeatureExtractor(
        strategy='cnn',
        include_color=False,
        include_texture=True,
        include_edges=False,
        cnn_model='resnet18',
        target_size=(224, 224)
    )
    assert extractor.strategy == 'cnn'
    assert extractor.include_color is False
    assert extractor.include_texture is True
    assert extractor.include_edges is False
    assert extractor.cnn_model == 'resnet18'
    assert extractor.target_size == (224, 224)

def test_initialization_invalid():
    """Test initialization with invalid parameters."""
    with pytest.raises(ValueError):
        ImageFeatureExtractor(strategy='invalid_strategy')
```

#### 2.1.2 Image Loading Tests

- Test loading images from file paths
- Test loading images from arrays
- Test loading images from tensors
- Test loading images with different formats (JPEG, PNG, etc.)
- Test loading images with different color modes (RGB, grayscale)
- Test loading images with different sizes
- Test handling of invalid images

```python
def test_load_images_from_paths(sample_image_paths):
    """Test loading images from file paths."""
    extractor = ImageFeatureExtractor()
    images = extractor._load_images(sample_image_paths)
    assert len(images) == len(sample_image_paths)
    # Verify image properties

def test_load_images_from_arrays(sample_image_arrays):
    """Test loading images from arrays."""
    extractor = ImageFeatureExtractor()
    images = extractor._load_images(sample_image_arrays)
    assert len(images) == len(sample_image_arrays)
    # Verify image properties

def test_load_images_with_resizing():
    """Test loading images with resizing."""
    extractor = ImageFeatureExtractor(target_size=(100, 100))
    images = extractor._load_images(sample_image_paths)
    for img in images:
        assert img.shape[:2] == (100, 100)
```

#### 2.1.3 Feature Extraction Tests

- Test basic feature extraction
  - Dimensional features
  - Color features
  - Texture features
  - Edge features
- Test CNN-based feature extraction
  - Different models
  - Different layers
- Test custom feature extraction
- Test combined strategies

```python
def test_extract_dimensional_features(sample_images):
    """Test extraction of dimensional features."""
    extractor = ImageFeatureExtractor(include_color=False, include_texture=False, include_edges=False)
    features = extractor.fit_transform(sample_images)
    # Verify dimensional features are present and correct

def test_extract_color_features(sample_images):
    """Test extraction of color features."""
    extractor = ImageFeatureExtractor(include_texture=False, include_edges=False)
    features = extractor.fit_transform(sample_images)
    # Verify color features are present and correct

def test_extract_texture_features(sample_images):
    """Test extraction of texture features."""
    extractor = ImageFeatureExtractor(include_color=False, include_edges=False)
    features = extractor.fit_transform(sample_images)
    # Verify texture features are present and correct

def test_extract_edge_features(sample_images):
    """Test extraction of edge features."""
    extractor = ImageFeatureExtractor(include_color=False, include_texture=False)
    features = extractor.fit_transform(sample_images)
    # Verify edge features are present and correct

def test_extract_cnn_features(sample_images):
    """Test extraction of CNN-based features."""
    extractor = ImageFeatureExtractor(strategy='cnn', cnn_model='resnet18')
    features = extractor.fit_transform(sample_images)
    # Verify CNN features are present and correct

def test_extract_custom_features(sample_images):
    """Test extraction of custom features."""
    def custom_extractor(images):
        # Custom feature extraction logic
        return {'custom_feature': [1.0] * len(images)}
    
    extractor = ImageFeatureExtractor(strategy='custom', custom_extractor=custom_extractor)
    features = extractor.fit_transform(sample_images)
    # Verify custom features are present and correct
```

#### 2.1.4 Backend Compatibility Tests

- Test with NumPy backend
- Test with PyTorch backend
- Test with MLX backend
- Test automatic backend detection

```python
def test_numpy_backend(sample_images):
    """Test with NumPy backend."""
    extractor = ImageFeatureExtractor(backend='numpy')
    features = extractor.fit_transform(sample_images)
    # Verify features are correct with NumPy backend

def test_torch_backend(sample_images):
    """Test with PyTorch backend."""
    extractor = ImageFeatureExtractor(backend='torch')
    features = extractor.fit_transform(sample_images)
    # Verify features are correct with PyTorch backend

def test_mlx_backend(sample_images):
    """Test with MLX backend."""
    extractor = ImageFeatureExtractor(backend='mlx')
    features = extractor.fit_transform(sample_images)
    # Verify features are correct with MLX backend

def test_auto_backend_detection(sample_images):
    """Test automatic backend detection."""
    extractor = ImageFeatureExtractor()
    features = extractor.fit_transform(sample_images)
    # Verify backend was correctly detected and features are correct
```

### 2.2 Integration Tests

Integration tests will verify that the `ImageFeatureExtractor` works correctly with other components of the EmberHarmony system.

#### 2.2.1 Pipeline Integration Tests

- Test integration with scikit-learn pipelines
- Test integration with other EmberHarmony extractors
- Test integration with EmberHarmony models

```python
def test_sklearn_pipeline_integration():
    """Test integration with scikit-learn pipelines."""
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    
    extractor = ImageFeatureExtractor()
    pipeline = Pipeline([
        ('features', extractor),
        ('classifier', RandomForestClassifier())
    ])
    
    # Test pipeline with sample data
    pipeline.fit(sample_images, sample_labels)
    predictions = pipeline.predict(test_images)
    # Verify predictions

def test_integration_with_column_extractor():
    """Test integration with ColumnFeatureExtractor."""
    from emberharmony.features.column_feature_extraction import ColumnFeatureExtractor
    
    # Create a DataFrame with image paths and other columns
    df = pd.DataFrame({
        'image_path': sample_image_paths,
        'numeric_feature': [1.0, 2.0, 3.0],
        'categorical_feature': ['A', 'B', 'C']
    })
    
    # Extract features from images
    image_extractor = ImageFeatureExtractor()
    image_features = image_extractor.fit_transform(df['image_path'])
    
    # Extract features from other columns
    column_extractor = ColumnFeatureExtractor()
    column_features = column_extractor.fit_transform(df[['numeric_feature', 'categorical_feature']])
    
    # Combine features
    combined_features = pd.concat([image_features, column_features], axis=1)
    # Verify combined features
```

#### 2.2.2 Backend Integration Tests

- Test integration with the ops abstraction layer
- Test consistency across different backends

```python
def test_ops_integration():
    """Test integration with the ops abstraction layer."""
    from emberharmony import ops
    
    extractor = ImageFeatureExtractor()
    features = extractor.fit_transform(sample_images)
    
    # Verify features can be processed with ops functions
    processed_features = ops.normalize(ops.convert_to_tensor(features))
    # Verify processed features

def test_backend_consistency():
    """Test consistency across different backends."""
    # Extract features with different backends
    numpy_extractor = ImageFeatureExtractor(backend='numpy')
    numpy_features = numpy_extractor.fit_transform(sample_images)
    
    torch_extractor = ImageFeatureExtractor(backend='torch')
    torch_features = torch_extractor.fit_transform(sample_images)
    
    # Convert to common format for comparison
    numpy_features_array = numpy_features.values
    torch_features_array = torch_features.values
    
    # Verify features are consistent across backends
    np.testing.assert_allclose(numpy_features_array, torch_features_array, rtol=1e-5, atol=1e-5)
```

### 2.3 Performance Tests

Performance tests will verify that the `ImageFeatureExtractor` performs efficiently with various dataset sizes.

#### 2.3.1 Scalability Tests

- Test with small datasets (10-100 images)
- Test with medium datasets (100-1000 images)
- Test with large datasets (1000+ images)

```python
def test_small_dataset_performance():
    """Test performance with small datasets."""
    import time
    
    extractor = ImageFeatureExtractor()
    start_time = time.time()
    features = extractor.fit_transform(small_dataset)
    end_time = time.time()
    
    # Verify performance is acceptable
    assert end_time - start_time < 5.0  # Should process in less than 5 seconds

def test_medium_dataset_performance():
    """Test performance with medium datasets."""
    import time
    
    extractor = ImageFeatureExtractor()
    start_time = time.time()
    features = extractor.fit_transform(medium_dataset)
    end_time = time.time()
    
    # Verify performance is acceptable
    assert end_time - start_time < 30.0  # Should process in less than 30 seconds

def test_large_dataset_performance():
    """Test performance with large datasets."""
    import time
    
    extractor = ImageFeatureExtractor()
    start_time = time.time()
    features = extractor.fit_transform(large_dataset)
    end_time = time.time()
    
    # Verify performance is acceptable
    assert end_time - start_time < 300.0  # Should process in less than 5 minutes
```

#### 2.3.2 Memory Usage Tests

- Test memory usage with different dataset sizes
- Test memory usage with different feature extraction strategies

```python
def test_memory_usage():
    """Test memory usage with different dataset sizes and strategies."""
    import tracemalloc
    
    # Test basic strategy
    tracemalloc.start()
    extractor = ImageFeatureExtractor(strategy='basic')
    extractor.fit_transform(medium_dataset)
    basic_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    
    # Test CNN strategy
    tracemalloc.start()
    extractor = ImageFeatureExtractor(strategy='cnn')
    extractor.fit_transform(medium_dataset)
    cnn_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    
    # Verify memory usage is acceptable
    assert basic_memory < 1e9  # Less than 1GB for basic strategy
    assert cnn_memory < 2e9  # Less than 2GB for CNN strategy
```

### 2.4 Edge Case Tests

Edge case tests will verify that the `ImageFeatureExtractor` handles unusual or problematic inputs correctly.

#### 2.4.1 Input Validation Tests

- Test with empty input
- Test with invalid file paths
- Test with corrupted images
- Test with images of different sizes
- Test with grayscale images
- Test with RGBA images

```python
def test_empty_input():
    """Test with empty input."""
    extractor = ImageFeatureExtractor()
    with pytest.raises(ValueError):
        extractor.fit_transform([])

def test_invalid_file_paths():
    """Test with invalid file paths."""
    extractor = ImageFeatureExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.fit_transform(['nonexistent_image.jpg'])

def test_corrupted_images():
    """Test with corrupted images."""
    extractor = ImageFeatureExtractor()
    # Create a corrupted image file
    with open('corrupted_image.jpg', 'wb') as f:
        f.write(b'not an image')
    
    # Test with a mix of valid and corrupted images
    result = extractor.fit_transform(['valid_image.jpg', 'corrupted_image.jpg'])
    # Verify that valid images are processed and corrupted ones are handled gracefully

def test_different_size_images():
    """Test with images of different sizes."""
    extractor = ImageFeatureExtractor()
    # Create images of different sizes
    result = extractor.fit_transform(['small_image.jpg', 'large_image.jpg'])
    # Verify that images are properly resized or features are normalized

def test_grayscale_images():
    """Test with grayscale images."""
    extractor = ImageFeatureExtractor()
    result = extractor.fit_transform(['grayscale_image.jpg'])
    # Verify that grayscale images are properly handled

def test_rgba_images():
    """Test with RGBA images."""
    extractor = ImageFeatureExtractor()
    result = extractor.fit_transform(['rgba_image.png'])
    # Verify that RGBA images are properly handled
```

## 3. Test Fixtures

### 3.1 Sample Images

Create a set of sample images for testing:

- Small set of diverse images (different sizes, formats, content)
- Images with known properties for verification
- Synthetic images with controlled features

```python
@pytest.fixture
def sample_image_paths():
    """Create sample image paths for testing."""
    # Create or locate sample images
    return ['tests/data/images/sample1.jpg', 'tests/data/images/sample2.png', 'tests/data/images/sample3.jpg']

@pytest.fixture
def sample_image_arrays():
    """Create sample image arrays for testing."""
    # Create synthetic image arrays
    import numpy as np
    
    # Create a red square
    red_square = np.zeros((100, 100, 3), dtype=np.uint8)
    red_square[:, :, 0] = 255
    
    # Create a green square
    green_square = np.zeros((100, 100, 3), dtype=np.uint8)
    green_square[:, :, 1] = 255
    
    # Create a blue square
    blue_square = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_square[:, :, 2] = 255
    
    return [red_square, green_square, blue_square]

@pytest.fixture
def sample_images(sample_image_paths):
    """Load sample images for testing."""
    from PIL import Image
    return [np.array(Image.open(path)) for path in sample_image_paths]
```

### 3.2 Test Datasets

Create datasets of different sizes for performance testing:

```python
@pytest.fixture
def small_dataset():
    """Create a small dataset for testing."""
    # Generate or locate a small set of images
    return ['tests/data/images/small_dataset/img{:03d}.jpg'.format(i) for i in range(10)]

@pytest.fixture
def medium_dataset():
    """Create a medium dataset for testing."""
    # Generate or locate a medium set of images
    return ['tests/data/images/medium_dataset/img{:03d}.jpg'.format(i) for i in range(100)]

@pytest.fixture
def large_dataset():
    """Create a large dataset for testing."""
    # Generate or locate a large set of images
    return ['tests/data/images/large_dataset/img{:03d}.jpg'.format(i) for i in range(1000)]
```

## 4. Test Execution

### 4.1 Test Environment Setup

- Set up a test environment with all required dependencies
- Create test data directories and sample images
- Configure pytest for the test suite

### 4.2 Test Execution Process

1. Run unit tests first to verify basic functionality
2. Run integration tests to verify compatibility with other components
3. Run performance tests to verify efficiency
4. Run edge case tests to verify robustness

### 4.3 Continuous Integration

- Configure CI pipeline to run tests automatically
- Set up test coverage reporting
- Define acceptance criteria for test results

## 5. Test Reporting

### 5.1 Test Coverage

- Aim for at least 90% code coverage
- Ensure all critical paths are covered
- Identify and document any untested code

### 5.2 Test Results

- Generate detailed test reports
- Track test failures and fixes
- Maintain a history of test results

## 6. Test Maintenance

### 6.1 Test Data Management

- Store test data in a version-controlled repository
- Document the purpose and properties of test data
- Update test data as needed for new features or bug fixes

### 6.2 Test Suite Updates

- Update tests when the `ImageFeatureExtractor` is modified
- Add new tests for new features or bug fixes
- Refactor tests to improve maintainability

## 7. Acceptance Criteria

The `ImageFeatureExtractor` implementation will be considered tested and ready for release when:

1. All unit tests pass
2. All integration tests pass
3. Performance meets the defined benchmarks
4. Edge cases are handled correctly
5. Test coverage meets the defined threshold
6. Documentation is complete and accurate