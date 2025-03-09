# Testing Documentation

This directory contains documentation related to testing procedures and test plans for the Liquid Neural Network project.

## Available Documentation

- [**Test Stride Ware CFC Test Plan**](test_stride_ware_cfc_test_plan.md): Test plan for the Stride Ware Continuous Feedback Control component

## Testing Approach

The project follows a comprehensive testing approach that includes:

1. **Unit Testing**: Testing individual components in isolation
2. **Integration Testing**: Testing the interaction between components
3. **System Testing**: Testing the entire system as a whole
4. **Performance Testing**: Testing the performance of the system under various conditions

## Running Tests

The project includes several test scripts that can be used to run tests:

- `run_tests.py`: Runs all tests in the project
- `run_purification_tests.py`: Runs tests for the purified implementation
- `run_purification_tests_v2.py`: Runs tests for the updated purified implementation

To run all tests:

```bash
python tests/run_tests.py
```

To run specific tests:

```bash
python -m unittest tests.test_backend_auto_selection
```

## Test Coverage

The project aims for high test coverage, with a focus on testing:

1. **Backend Compatibility**: Ensuring that code works with all supported backends (MLX, PyTorch, NumPy)
2. **Memory Efficiency**: Verifying that operations minimize memory usage
3. **Scalability**: Testing with large datasets to ensure scalability
4. **Correctness**: Verifying that results are correct across different backends and configurations

## Contributing Tests

When contributing new features or fixing bugs, it's important to include tests that:

1. Verify the correctness of the implementation
2. Test edge cases and error conditions
3. Ensure compatibility with all supported backends
4. Measure performance where relevant

## Continuous Integration

The project uses continuous integration to automatically run tests on each commit, ensuring that changes don't break existing functionality.