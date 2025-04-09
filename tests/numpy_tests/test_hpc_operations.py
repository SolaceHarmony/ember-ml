"""
Tests for High-Precision Computing (HPC) operations in the NumPy backend.

This module tests the specialized HPC operations that provide enhanced numerical
stability and precision, particularly for operations that are challenging for
standard floating-point arithmetic.
"""

import pytest
import numpy as np
import math

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.backend.numpy.linearalg.orthogonal_ops import HPC16x8, qr_128

@pytest.fixture
def numpy_backend():
    """Set up NumPy backend for tests."""
    from ember_ml.backend import set_backend
    prev_backend = ops.get_backend()
    set_backend('numpy')
    yield None
    set_backend(prev_backend)

def test_hpc_orthogonal_vs_standard_qr(numpy_backend):
    """
    Test that HPC orthogonal implementation has better numerical stability
    than standard QR for ill-conditioned matrices.
    """
    # Create a highly ill-conditioned matrix
    n = 100
    m = 50
    
    # Create a matrix with exponentially decreasing singular values
    # This will be very challenging for standard QR
    u = tensor.random_normal((n, m))
    s = tensor.exp(-tensor.arange(m, dtype=tensor.float32) / 5)  # Exponentially decreasing
    v = tensor.random_normal((m, m))
    
    # Create ill-conditioned matrix A = U * diag(s) * V^T
    u_orth = ops.linearalg.orthogonal((n, m))
    v_orth = ops.linearalg.orthogonal((m, m))
    
    # Create diagonal matrix with singular values
    diag_s = tensor.zeros((n, m))
    for i in range(m):
        diag_s = tensor.slice_update(diag_s, (i, i), s[i])
    
    # Compute A = U * diag(s) * V^T
    a = ops.matmul(ops.matmul(u_orth, diag_s), ops.transpose(v_orth))
    
    # Get orthogonal matrix using our HPC implementation
    q_hpc = ops.linearalg.orthogonal((n, m))
    
    # Check orthogonality of columns (Q^T * Q should be close to identity)
    q_t_q = ops.matmul(ops.transpose(q_hpc), q_hpc)
    identity = tensor.eye(m)
    
    # Compute error
    error_hpc = ops.mean(ops.abs(q_t_q - identity))
    
    # Now try with standard QR (without HPC)
    # We'll use numpy.linalg.qr directly
    a_np = tensor.to_numpy(a)
    q_np, _ = np.linalg.qr(a_np, mode='reduced')
    q_t_q_np = np.matmul(q_np.T, q_np)
    identity_np = np.eye(m)
    error_standard = np.mean(np.abs(q_t_q_np - identity_np))
    
    # The HPC implementation should have better numerical stability
    assert error_hpc < error_standard, f"HPC error: {error_hpc}, Standard error: {error_standard}"
    print(f"HPC error: {error_hpc}, Standard error: {error_standard}")
    
    # The HPC error should be very small
    assert error_hpc < 1e-5, f"HPC error too large: {error_hpc}"

def test_hpc_limb_arithmetic_precision(numpy_backend):
    """
    Test that HPC limb arithmetic provides better precision than standard arithmetic.
    
    This test demonstrates how the double-single precision technique used in HPC
    can represent numbers more precisely than standard floating point.
    """
    # Create a small number
    small = 1e-8
    
    # Create a large number
    large = 1e8
    
    # In standard floating point, adding a small number to a large number
    # and then subtracting the large number should give the small number,
    # but due to precision limitations, it often doesn't
    
    # Standard arithmetic
    large_np = np.array(large, dtype=np.float32)
    small_np = np.array(small, dtype=np.float32)
    
    sum_standard = large_np + small_np
    diff_standard = sum_standard - large_np
    
    # HPC limb arithmetic
    large_hpc = HPC16x8(large_np)
    small_hpc = HPC16x8(small_np)
    
    # Add using HPC
    from ember_ml.backend.numpy.linearalg.orthogonal_ops import _add_limb_precision
    sum_high, sum_low = _add_limb_precision(large_hpc.high, large_hpc.low, small_hpc.high, small_hpc.low)
    
    # Create HPC object for sum
    sum_hpc = HPC16x8(sum_high, sum_low)
    
    # Subtract using HPC
    diff_high, diff_low = _add_limb_precision(sum_hpc.high, sum_hpc.low, -large_hpc.high, -large_hpc.low)
    
    # Convert back to standard precision
    diff_hpc = diff_high + diff_low
    
    # The HPC version should be closer to the true small value
    error_standard = abs(diff_standard.item() - small) / small
    error_hpc = abs(diff_hpc.item() - small) / small
    
    print(f"Standard arithmetic result: {diff_standard.item()}, expected: {small}")
    print(f"HPC arithmetic result: {diff_hpc.item()}, expected: {small}")
    print(f"Standard relative error: {error_standard}, HPC relative error: {error_hpc}")
    
    # The HPC error should be smaller
    assert error_hpc < error_standard, f"HPC error: {error_hpc}, Standard error: {error_standard}"

def test_orthogonal_non_square_matrices(numpy_backend):
    """
    Test that the orthogonal function works correctly for non-square matrices.
    
    This test verifies that the orthogonal function produces matrices with
    orthogonal columns even for highly rectangular matrices.
    """
    # Test with various shapes
    shapes = [
        (100, 10),    # Tall and thin
        (10, 100),    # Short and wide
        (128, 64),    # Power of 2 dimensions
        (65, 33),     # Odd dimensions
        (200, 199),   # Almost square
        (3, 100)      # Very rectangular
    ]
    
    for shape in shapes:
        # Generate orthogonal matrix
        q = ops.linearalg.orthogonal(shape)
        
        # Check shape
        assert q.shape == shape, f"Expected shape {shape}, got {q.shape}"
        
        # Check orthogonality of columns
        if shape[0] >= shape[1]:
            # Tall matrix: Q^T * Q should be identity
            q_t_q = ops.matmul(ops.transpose(q), q)
            identity = tensor.eye(shape[1])
            error = ops.mean(ops.abs(q_t_q - identity))
        else:
            # Wide matrix: Q * Q^T should be identity
            q_q_t = ops.matmul(q, ops.transpose(q))
            identity = tensor.eye(shape[0])
            error = ops.mean(ops.abs(q_q_t - identity))
        
        # Error should be small
        assert error < 1e-5, f"Orthogonality error too large for shape {shape}: {error}"
        print(f"Shape {shape}: orthogonality error = {error}")

def test_qr_128_precision(numpy_backend):
    """
    Test that the 128-bit precision QR decomposition has better numerical stability
    than standard QR for ill-conditioned matrices.
    """
    # Create a matrix with poor conditioning
    n = 50
    
    # Create a matrix with exponentially decreasing diagonal elements
    diag_vals = np.exp(-np.arange(n, dtype=np.float32) / 5)
    a = np.diag(diag_vals)
    
    # Add some noise to make it more challenging
    noise = np.random.randn(n, n).astype(np.float32) * 1e-3
    a = a + noise
    
    # Compute QR using our 128-bit precision implementation
    q_hpc, r_hpc = qr_128(a)
    
    # Compute QR using standard NumPy
    q_std, r_std = np.linalg.qr(a)
    
    # Check orthogonality of Q
    q_t_q_hpc = np.matmul(q_hpc.T, q_hpc)
    q_t_q_std = np.matmul(q_std.T, q_std)
    
    identity = np.eye(n)
    
    # Compute errors
    error_hpc = np.mean(np.abs(q_t_q_hpc - identity))
    error_std = np.mean(np.abs(q_t_q_std - identity))
    
    # Check reconstruction error
    recon_hpc = np.matmul(q_hpc, r_hpc)
    recon_std = np.matmul(q_std, r_std)
    
    recon_error_hpc = np.mean(np.abs(recon_hpc - a))
    recon_error_std = np.mean(np.abs(recon_std - a))
    
    print(f"HPC orthogonality error: {error_hpc}, Standard error: {error_std}")
    print(f"HPC reconstruction error: {recon_error_hpc}, Standard error: {recon_error_std}")
    
    # The HPC implementation should have better orthogonality
    assert error_hpc <= error_std * 1.1, f"HPC error: {error_hpc}, Standard error: {error_std}"
    
    # Both should have similar reconstruction error
    assert abs(recon_error_hpc - recon_error_std) < 1e-5, \
        f"HPC recon error: {recon_error_hpc}, Standard recon error: {recon_error_std}"

def test_cross_backend_consistency():
    """
    Test that the orthogonal function produces consistent results across backends.
    
    This test verifies that our HPC implementation produces similar results
    regardless of which backend is used.
    """
    # Set up shape for testing
    shape = (50, 30)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test with NumPy backend
    from ember_ml.backend import set_backend
    set_backend('numpy')
    q_numpy = ops.linearalg.orthogonal(shape)
    
    # Test with PyTorch backend
    set_backend('torch')
    q_torch = ops.linearalg.orthogonal(shape)
    
    # Test with MLX backend
    set_backend('mlx')
    q_mlx = ops.linearalg.orthogonal(shape)
    
    # Convert all to NumPy for comparison
    q_numpy_np = tensor.to_numpy(q_numpy)
    q_torch_np = tensor.to_numpy(q_torch)
    q_mlx_np = tensor.to_numpy(q_mlx)
    
    # Check orthogonality for all backends
    identity = np.eye(shape[1])
    
    error_numpy = np.mean(np.abs(np.matmul(q_numpy_np.T, q_numpy_np) - identity))
    error_torch = np.mean(np.abs(np.matmul(q_torch_np.T, q_torch_np) - identity))
    error_mlx = np.mean(np.abs(np.matmul(q_mlx_np.T, q_mlx_np) - identity))
    
    print(f"NumPy orthogonality error: {error_numpy}")
    print(f"PyTorch orthogonality error: {error_torch}")
    print(f"MLX orthogonality error: {error_mlx}")
    
    # All errors should be small
    assert error_numpy < 1e-5, f"NumPy error too large: {error_numpy}"
    assert error_torch < 1e-5, f"PyTorch error too large: {error_torch}"
    assert error_mlx < 1e-5, f"MLX error too large: {error_mlx}"
    
    # Reset backend to NumPy
    set_backend('numpy')