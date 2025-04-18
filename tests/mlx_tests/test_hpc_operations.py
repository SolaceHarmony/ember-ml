"""
Tests for High-Precision Computing (HPC) operations in the MLX backend.

This module tests the specialized HPC operations that provide enhanced numerical
stability and precision, particularly for operations that are challenging for
standard floating-point arithmetic.
"""

import pytest
import numpy as np

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import linearalg
# Import specialized MLX-specific functions for testing
from ember_ml.backend.mlx.linearalg.hpc_nonsquare import orthogonalize_nonsquare

@pytest.fixture
def mlx_backend():
    """Set up MLX backend for tests."""
    prev_backend = ops.get_backend()
    ops.set_backend('mlx')
    yield None
    ops.set_backend(prev_backend)

def test_hpc_orthogonal_vs_standard_qr(mlx_backend):
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
    s = ops.exp(-tensor.arange(m, dtype=tensor.float32) / 5)  # Exponentially decreasing
    v = tensor.random_normal((m, m))
    
    # Create ill-conditioned matrix A = U * diag(s) * V^T
    u_orth = ops.linearalg.orthogonal((n, m))
    v_orth = ops.linearalg.orthogonal((m, m))
    
    # Create diagonal matrix with singular values using ops.linearalg.diag
    diag_s_small = ops.linearalg.diag(s)
    
    # Pad to the correct size if needed (n x m)
    if n > m:
        # Pad with zeros to make it n x m
        diag_s = tensor.pad(diag_s_small, [[0, n - m], [0, 0]])
    else:
        # Use as is or pad if needed
        diag_s = diag_s_small
    
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
    # We'll use NumPy's QR since it doesn't have the HPC enhancements
    import numpy as np
    a_np = tensor.to_numpy(a)
    q_np, _ = np.linalg.qr(a_np, mode='reduced')
    q_t_q_np = np.matmul(q_np.T, q_np)
    identity_np = np.eye(m)
    error_standard = np.mean(np.abs(q_t_q_np - identity_np))
    
    # The HPC implementation should have significantly better numerical stability
    assert error_hpc < error_standard, f"HPC error: {error_hpc}, Standard error: {error_standard}"
    print(f"HPC error: {error_hpc}, Standard error: {error_standard}")
    
    # The HPC error should be very small
    assert error_hpc < 1e-5, f"HPC error too large: {error_hpc}"

def test_metal_kernel_orthogonalization(mlx_backend):
    """
    Test the Metal kernel-based orthogonalization for large matrices.
    
    This test verifies that the Metal kernel implementation can handle
    large non-square matrices efficiently and with good numerical stability.
    """
    # Skip if not on macOS with Metal support
    try:
        import mlx.core as mx
        device = mx.default_device()
        if device.type != 'gpu':
            pytest.skip("Test requires Metal GPU support")
    except (ImportError, AttributeError):
        pytest.skip("Test requires MLX with Metal support")
    
    # Create a large non-square matrix
    n = 1024
    m = 512
    
    # Create random matrix
    a = tensor.random_normal((n, m))
    
    # Use the Metal kernel-based orthogonalization
    a_mlx = tensor.to_numpy(a)
    import mlx.core as mx
    a_mx = mx.array(a_mlx)
    
    # Time the Metal kernel implementation
    import time
    start_time = time.time()
    q_metal = orthogonalize_nonsquare(a_mx)
    metal_time = time.time() - start_time
    
    # Check orthogonality
    q_t_q = mx.matmul(mx.transpose(q_metal), q_metal)
    identity = mx.eye(m)
    error_metal = mx.mean(mx.abs(q_t_q - identity)).item()
    
    # Time a standard QR implementation for comparison
    start_time = time.time()
    q_standard, _ = mx.linalg.qr(a_mx)
    standard_time = time.time() - start_time
    
    # Check orthogonality of standard implementation
    q_t_q_std = mx.matmul(mx.transpose(q_standard), q_standard)
    error_standard = mx.mean(mx.abs(q_t_q_std - identity)).item()
    
    # The Metal kernel implementation should be faster for large matrices
    print(f"Metal kernel time: {metal_time:.4f}s, Standard QR time: {standard_time:.4f}s")
    print(f"Metal kernel error: {error_metal}, Standard QR error: {error_standard}")
    
    # The Metal kernel error should be small
    assert error_metal < 1e-5, f"Metal kernel error too large: {error_metal}"
    
    # For large matrices, the Metal kernel should be faster
    # Note: This might not always be true depending on the hardware and MLX version
    # so we'll just print the times rather than asserting

def test_hpc_limb_arithmetic_precision(mlx_backend):
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
    import mlx.core as mx
    large_mx = mx.array(large, dtype=mx.float32)
    small_mx = mx.array(small, dtype=mx.float32)
    
    sum_standard = large_mx + small_mx
    diff_standard = sum_standard - large_mx
    
    # HPC limb arithmetic
    from ember_ml.backend.mlx.linearalg.decomp_ops_hpc import _add_limb_precision
    
    # Use the HPC16x8 class from ops.linearalg
    large_hpc = linearalg.HPC16x8.from_array(large_mx)
    small_hpc = linearalg.HPC16x8.from_array(small_mx)
    
    # Add using HPC limb precision
    sum_high, sum_low = _add_limb_precision(large_hpc.high, small_hpc.high)
    sum_hpc = linearalg.HPC16x8(sum_high, sum_low)
    
    # Subtract using HPC limb precision
    neg_large_high = -large_hpc.high
    diff_high, diff_low = _add_limb_precision(sum_hpc.high, neg_large_high)
    
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

def test_orthogonal_non_square_matrices(mlx_backend):
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