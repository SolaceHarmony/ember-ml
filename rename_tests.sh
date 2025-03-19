#!/bin/bash

# Script to rename test files according to the naming convention

# Frontend tensor tests
mv tests/test_ember_tensor_comprehensive.py tests/test_nn_tensor_comprehensive.py
mv tests/test_ember_tensor_operations.py tests/test_nn_tensor_operations.py
mv tests/test_ember_tensor_size.py tests/test_nn_tensor_size.py
mv tests/test_device.py tests/test_nn_tensor_device_simple.py
mv tests/test_dtype_usage.py tests/test_nn_tensor_dtype_usage.py

# Backend-specific tests
mv tests/test_mlx_cast.py tests/test_backend_mlx_tensor_cast.py
mv tests/test_mlx_cast_simple.py tests/test_backend_mlx_tensor_cast_simple.py
mv tests/test_mlx_fix.py tests/test_backend_mlx_tensor_fix.py
mv tests/test_mlx_slice.py tests/test_backend_mlx_tensor_slice.py
# Check if this is a duplicate before renaming
if [ -f tests/test_mlx_tensor_slice.py ] && [ ! -f tests/test_backend_mlx_tensor_slice.py ]; then
    mv tests/test_mlx_tensor_slice.py tests/test_backend_mlx_tensor_slice.py
fi
mv tests/test_device_numpy.py tests/test_backend_numpy_device.py
mv tests/test_device_torch.py tests/test_backend_torch_device.py
mv tests/test_numpy_tensor_ops.py tests/test_backend_numpy_tensor_ops.py
mv tests/test_numpy_tensor_ops_direct.py tests/test_backend_numpy_tensor_ops_direct.py
mv tests/test_simple_mlx.py tests/test_backend_mlx_simple.py
mv tests/test_ember_backend_simple.py tests/test_backend_simple.py

echo "File renaming completed."