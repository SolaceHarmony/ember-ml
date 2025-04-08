"""
MLX solver operations for ember_ml.

This module provides MLX implementations of matrix decomposition operations.
"""
from typing import Union, Tuple, Literal, Optional, List
import mlx.core as mx

# Import from tensor_ops
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.linearalg.decomp_ops_hpc import HPC16x8 # Corrected path

dtype_obj = MLXDType()

def _update_array(arr: mx.array, indices: Union[int, Tuple[int, ...], slice], value: mx.array) -> mx.array:
    """Helper function for MLX array updates."""
    # Create copy and update
    result = mx.array(arr)
    result[indices] = value
    return result

def _slice_indices_to_array(indices: Union[List[int], List[List[int]]]) -> mx.array:
    """Convert Python list indices to MLX array."""
    return mx.array(indices, dtype=mx.int32)

def _to_sorted_array(tensor: mx.array, reverse: bool = False) -> mx.array:
    """Convert MLX tensor to sorted array safely."""
    # Convert to numpy, sort, and back to MLX
    import numpy as np
    np_array = np.array(tensor)
    sorted_array = np.sort(np_array)
    if reverse:
        sorted_array = sorted_array[::-1]
    return mx.array(sorted_array)

def _convert_to_float32(arr: Union[mx.array, 'HPC16x8']) -> mx.array:
    """Convert array or HPC object to float32 MLX array."""
    if isinstance(arr, HPC16x8):
        return arr.to_float32()
    return mx.array(arr, dtype=mx.float32)

def _convert_hpc_result(matrix_hpc: HPC16x8) -> Tuple[mx.array, mx.array]:
    """Convert HPC QR result to MLX arrays."""
    q_hpc, r_hpc = matrix_hpc.qr()
    if isinstance(q_hpc, HPC16x8):
        q = q_hpc.to_float32()
        r = _convert_to_float32(r_hpc)  # Use helper for consistency
    else:
        q = _convert_to_float32(q_hpc)
        r = _convert_to_float32(r_hpc)
    return q, r

def is_spd(A: mx.array) -> bool:
    """
    Check if matrix is symmetric positive definite.
    
    Args:
        A: Input matrix to check
        
    Returns:
        Boolean indicating if matrix is SPD
    """
    # Check for symmetry using mx.abs and mx.all
    diff = mx.subtract(A, mx.transpose(A))
    abs_diff = mx.abs(diff)
    is_symmetric = mx.all(mx.less(abs_diff, mx.array(1e-6))).item()
    if not is_symmetric:
        return False
    
    # For performance and precision, we should use our optimized Metal kernels
    # instead of moving to CPU
    try:
        # Use our optimized Metal implementation
        n = A.shape[0]
        if n < 32:
            cholesky_standard(A)
        elif n < 128:
            mlx_cholesky_single_thread(A)
        else:
            block_size = min(32, max(16, n // 32))
            mlx_cholesky_block_based(A, block_size=block_size)
        return True
    except Exception:
        return False

def mlx_cholesky_single_thread(A: mx.array) -> mx.array:
    """
    Stable implementation of Cholesky decomposition using single-threaded Metal approach.
    
    Args:
        A: Input positive definite matrix
        
    Returns:
        Lower triangular matrix L such that L @ L.T = A
    """
    @mx.custom_function
    def _inner_impl(A_inner: mx.array) -> mx.array:
        # Define Metal kernel source - using single thread approach for maximum stability
        source = """
        // Single-threaded implementation for maximum numerical stability
        if (thread_position_in_grid.x == 0) {
            // Get matrix size
            uint n = A_shape[0];
            
            // Initialize upper triangle to zero
            for (uint i = 0; i < n; i++) {
                for (uint j = i+1; j < n; j++) {
                    out[i*n + j] = 0.0f;
                }
            }
            
            // Standard Cholesky algorithm with strict sequential processing
            for (uint j = 0; j < n; j++) {
                // Compute diagonal element with accumulator for better precision
                float diag_sum = 0.0f;
                for (uint k = 0; k < j; k++) {
                    float val = out[j*n + k];
                    diag_sum += val * val;
                }
                
                float diag_val = A[j*n + j] - diag_sum;
                // Ensure positive diagonal for numerical stability
                if (diag_val <= 1e-10f) {
                    diag_val = 1e-10f;
                }
                out[j*n + j] = sqrt(diag_val);
                
                // Now compute all elements below diagonal in this column
                for (uint i = j+1; i < n; i++) {
                    float sum = 0.0f;
                    for (uint k = 0; k < j; k++) {
                        sum += out[i*n + k] * out[j*n + k];
                    }
                    
                    float denom = out[j*n + j];
                    if (denom > 1e-10f) {
                        out[i*n + j] = (A[i*n + j] - sum) / denom;
                    } else {
                        out[i*n + j] = 0.0f;
                    }
                }
            }
        }
        """
        
        # Metal header with math functions
        header = """
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;
        """
        
        # Create the kernel
        kernel = mx.fast.metal_kernel(
            name="cholesky_kernel",
            input_names=["A"],
            output_names=["out"],
            source=source,
            header=header,
            ensure_row_contiguous=True
        )
        
        # Single thread for maximum stability
        grid = (1, 1, 1)
        threads = (1, 1, 1)
        
        # Run the kernel
        result = kernel(
            inputs=[A_inner],
            output_shapes=[A_inner.shape],
            output_dtypes=[A_inner.dtype],
            grid=grid,
            threadgroup=threads
        ) # type: ignore
        return result[0]
    
    # Call the inner implementation
    return _inner_impl(A) # type: ignore

def mlx_cholesky_block_based(A: mx.array, block_size: int = 16) -> mx.array:
    """
    Block-based Cholesky implementation for handling larger matrices efficiently.
    
    Args:
        A: Input positive definite matrix
        block_size: Size of blocks for tiled computation (default: 16)
        
    Returns:
        Lower triangular matrix L such that L @ L.T = A
    """
    @mx.custom_function
    def _inner_impl(A_inner: mx.array, block_size_inner: int) -> mx.array:
        n = A_inner.shape[0]
        
        # Define Metal kernel source for block-based approach
        source = """
        // Get thread ID and block size
        uint thread_id = thread_position_in_grid.x;
        uint n = A_shape[0];
        uint block_size = block_param[0];
        uint num_blocks = (n + block_size - 1) / block_size;
        uint num_threads = thread_count[0];  // Total number of threads
        
        // Process matrix in blocks
        for (uint k = 0; k < num_blocks; k++) {
            uint block_start = k * block_size;
            uint block_end = min(block_start + block_size, n);
            
            // Only thread 0 processes the diagonal block for stability
            if (thread_id == 0) {
                // Process diagonal block with standard Cholesky
                for (uint j = block_start; j < block_end; j++) {
                    // Compute diagonal element
                    float sum_diag = 0.0f;
                    for (uint p = 0; p < j; p++) {
                        sum_diag += out[j*n + p] * out[j*n + p];
                    }
                    
                    float diag_val = A[j*n + j] - sum_diag;
                    if (diag_val <= 1e-10f) {
                        diag_val = 1e-10f;
                    }
                    out[j*n + j] = sqrt(diag_val);
                    
                    // Compute off-diagonals in this column
                    for (uint i = j+1; i < block_end; i++) {
                        float sum = 0.0f;
                        for (uint p = 0; p < j; p++) {
                            sum += out[i*n + p] * out[j*n + p];
                        }
                        
                        float denom = out[j*n + j];
                        if (denom > 1e-10f) {
                            out[i*n + j] = (A[i*n + j] - sum) / denom;
                        } else {
                            out[i*n + j] = 0.0f;
                        }
                    }
                }
            }
            
            // Wait for diagonal block to complete
            threadgroup_barrier(mem_flags::mem_device);
            
            // Initialize upper triangles to zero (all threads participate)
            for (uint i = thread_id; i < n; i += num_threads) {
                for (uint j = i+1; j < n; j++) {
                    if ((i < block_start && j >= block_start && j < block_end) ||
                        (i >= block_start && i < block_end && j >= block_end)) {
                        out[i*n + j] = 0.0f;
                    }
                }
            }
            
            // Ensure zeros are set before computing elements
            threadgroup_barrier(mem_flags::mem_device);
            
            // Each thread processes a set of rows for remaining blocks
            for (uint row = thread_id; row < n; row += num_threads) {
                // Only process rows below the current block
                if (row >= block_end) {
                    // Update the row using the diagonal block
                    for (uint j = block_start; j < block_end; j++) {
                        float sum = 0.0f;
                        for (uint p = 0; p < j; p++) {
                            sum += out[row*n + p] * out[j*n + p];
                        }
                        
                        float denom = out[j*n + j];
                        if (denom > 1e-10f) {
                            out[row*n + j] = (A[row*n + j] - sum) / denom;
                        } else {
                            out[row*n + j] = 0.0f;
                        }
                    }
                }
            }
            
            // Wait for all updates before moving to next block
            threadgroup_barrier(mem_flags::mem_device);
        }
        """
        
        # Metal header with math functions
        header = """
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;
        """
        
        # Create the kernel
        kernel = mx.fast.metal_kernel(
            name="block_cholesky_kernel",
            input_names=["A", "block_param", "thread_count"],
            output_names=["out"],
            source=source,
            header=header,
            ensure_row_contiguous=True
        )
        
        # Use multiple threads but not too many to maintain stability
        num_threads = min(32, n)
        grid = (num_threads, 1, 1)
        threads = (num_threads, 1, 1)
        
        # Parameters: block size and thread count
        block_param = mx.array([block_size_inner], dtype=mx.uint32)
        thread_count = mx.array([num_threads], dtype=mx.uint32)
        
        # Run the kernel
        result = kernel(
            inputs=[A_inner, block_param, thread_count],
            output_shapes=[A_inner.shape],
            output_dtypes=[A_inner.dtype],
            grid=grid,
            threadgroup=threads
        ) # type: ignore
        return result[0]
    
    # Call the inner implementation
    return _inner_impl(A, block_size) # type: ignore

def cholesky_standard(a_array: mx.array) -> mx.array:
    """
    Standard Python implementation of Cholesky decomposition.
    
    Args:
        a_array: Input positive definite matrix as MLX array
        
    Returns:
        Lower triangular matrix L such that L @ L.T = A
    """
    # Get matrix dimensions
    n = a_array.shape[0]
    
    # Initialize the result matrix
    l = mx.zeros((n, n), dtype=a_array.dtype)
    
    # Compute the Cholesky decomposition
    for i in range(n):
        # Use direct integer calculation
        i_plus_1_int = i + 1
        for j in range(i_plus_1_int):
            if mx.equal(i, j):
                # Diagonal element
                s = mx.subtract(a_array[i, i], mx.sum(mx.square(l[i, :j])))
                if mx.less(s, mx.array(0)):
                    raise ValueError("Matrix is not positive definite")
                
                # Update array using helper function
                l = _update_array(l, (i, i), mx.sqrt(s))
            else:
                # Off-diagonal element
                s = mx.subtract(a_array[i, j], mx.sum(mx.multiply(l[i, :j], l[j, :j])))
                
                # Update array using helper function
                l = _update_array(l, (i, j), mx.divide(s, l[j, j]))
    
    return l

def cholesky(a: TensorLike) -> mx.array:
    """
    Compute the Cholesky decomposition of a positive definite matrix.
    
    This function provides multiple implementation strategies based on matrix size
    and device to optimize performance while maintaining numerical stability:
    
    1. Standard implementation for small matrices (n < 32) or when using CPU
    2. Single-threaded Metal implementation for medium matrices (32 <= n < 128)
    3. Block-based Metal implementation for large matrices (n >= 128)
    
    Args:
        a: Input positive definite matrix
    
    Returns:
        Lower triangular matrix L such that L @ L.T = A
    
    Raises:
        ValueError: If matrix is not positive definite
        
    Notes:
        For Metal operations, this uses optimized kernel implementations that
        provide significant performance improvements over the standard approach.
    """
    # Convert input to MLX array with float32 dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    # Get matrix dimensions
    n = a_array.shape[0]
    assert a_array.shape[1] == n, "Matrix must be square"
    # Determine the current device
    try:
        current_device = mx.default_device()
        device_type = current_device.type
        is_metal = device_type == 'gpu'
    except Exception:
        # If we can't determine the device type, assume it's not Metal
        is_metal = False
    
    # Choose implementation based on matrix size and device
    if not is_metal or n < 32:
        # For small matrices or CPU, use standard implementation
        return cholesky_standard(a_array)
    elif n < 128:
        # For medium matrices on Metal, use single-threaded implementation
        try:
            return mlx_cholesky_single_thread(a_array)
        except Exception as e:
            # Fall back to standard implementation on failure
            print(f"Metal Cholesky failed with error: {e}. Falling back to standard implementation.")
            return cholesky_standard(a_array)
    else:
        # For large matrices on Metal, use block-based implementation
        try:
            # Adjust block size based on matrix dimension
            block_size = min(32, max(16, n // 32))
            return mlx_cholesky_block_based(a_array, block_size=block_size)
        except Exception as e:
            # Try single-threaded implementation
            try:
                print(f"Block Cholesky failed with error: {e}. Trying single-threaded implementation.")
                return mlx_cholesky_single_thread(a_array)
            except Exception as e2:
                # Fall back to standard implementation
                print(f"Metal Cholesky failed with error: {e2}. Falling back to standard implementation.")
                return cholesky_standard(a_array)

def _solve_triangular(L: mx.array, b: mx.array, upper: bool = False) -> mx.array:
    """
    Solve a triangular system of equations Lx = b.
    
    Uses MLX operations throughout for GPU acceleration.
    
    Args:
        L: Lower triangular matrix
        b: Right-hand side vector/matrix
        upper: If True, L is assumed upper triangular
        
    Returns:
        Solution x to Lx = b
    """
    n = L.shape[0]
    x = mx.zeros_like(b)
    
    if not upper:
        # Forward substitution for lower triangular
        for i in range(n):
            # Compute b - Lx for known elements
            rhs = b[i]
            for j in range(i):
                rhs = mx.subtract(rhs, mx.multiply(L[i, j], x[j]))
            
            # Divide by diagonal element
            x = _update_array(x, i, mx.divide(rhs, L[i, i]))
    else:
        # Back substitution for upper triangular
        for i in range(n-1, -1, -1):
            rhs = b[i]
            for j in range(i+1, n):
                rhs = mx.subtract(rhs, mx.multiply(L[i, j], x[j]))
            x = _update_array(x, i, mx.divide(rhs, L[i, i]))
    
    return x

def cholesky_inv(a: TensorLike) -> mx.array:
    """
    Compute the inverse of a symmetric positive definite matrix using Cholesky decomposition.
    
    This implementation first computes the Cholesky decomposition, then solves
    systems of equations to compute the inverse. It uses MLX operations throughout
    for GPU acceleration where possible.
    
    Args:
        a: Input symmetric positive definite matrix
        
    Returns:
        Inverse of input matrix
        
    Raises:
        ValueError: If matrix is not positive definite
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    try:
        # Try MLX's native cholesky first
        L = mx.linalg.cholesky(a_array)
    except Exception as e:
        print(f"Native Cholesky failed: {e}, using custom implementation")
        # Fall back to our custom Cholesky
        L = cholesky(a_array)
    
    n = L.shape[0]
    identity = mx.eye(n, dtype=a_array.dtype)
    
    # Compute inverse by solving systems of equations
    # First solve L y = I for y
    y = mx.zeros((n, n), dtype=a_array.dtype)
    for i in range(n):
        y_col = _solve_triangular(L, identity[:, i])
        y = _update_array(y, slice(None), y_col)
    
    # Then solve L^T x = y for x
    x = mx.zeros((n, n), dtype=a_array.dtype)
    L_T = mx.transpose(L)
    for i in range(n):
        x_col = _solve_triangular(L_T, y[:, i], upper=True)
        x = _update_array(x, slice(None), x_col)
    
    return x

def svd(a: TensorLike, 
        full_matrices: bool = True, compute_uv: bool = True) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """
    Compute the singular value decomposition of a matrix.
    
    This implementation uses MLX's native svd where possible, falling back to
    our HPC implementation for numerically sensitive cases.
    
    Args:
        a: Input matrix
        full_matrices: If True, return full U and Vh matrices
        compute_uv: If True, compute U and Vh matrices
    
    Returns:
        If compute_uv is True, returns (U, S, Vh), otherwise returns S
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    try:
        # Try native MLX SVD first
        result = mx.linalg.svd(a_array, compute_uv=compute_uv)
        if compute_uv:
            u, s, vh = result
            if not full_matrices:
                # Truncate to reduced form if needed
                m, n = a_array.shape
                k = min(m, n)
                u = u[:, :k]
                vh = vh[:k, :]
            return u, s, vh
        return result
    except Exception as e:
        # Fall back to HPC implementation for problematic cases
        print(f"Native SVD failed: {e}, using HPC implementation")
        m, n = a_array.shape
        k = min(m, n)
        
        # Use eigvalsh for symmetric case which is more stable
        if m >= n:
            ata = mx.matmul(mx.transpose(a_array), a_array)
            s = mx.sqrt(mx.abs(mx.linalg.eigvalsh(ata)))
            s = _to_sorted_array(s, reverse=True)[:k]
            
            if compute_uv:
                # Use stable HPC implementation for eigenvectors
                matrix_hpc = HPC16x8.from_array(ata)
                _, v = matrix_hpc.eig()
                # Convert result to float32 MLX array
                v = _convert_to_float32(v)
                v = v[:, mx.argsort(-s)][:, :k]
                
                # Compute U stably
                u = mx.zeros((m, k), dtype=a_array.dtype)
                for i in range(k):
                    if mx.greater(s[i], mx.array(1e-10)):
                        u_col = mx.divide(mx.matmul(a_array, v[:, i]), s[i])
                        u = _update_array(u, slice(None), u_col.reshape(-1, 1))
                    else:
                        u_col = mx.zeros((m,), dtype=a_array.dtype)
                        u_col = _update_array(u_col, i % m, mx.array(1.0))
                        u = _update_array(u, slice(None), u_col.reshape(-1, 1))
                
                if full_matrices and m > k:
                    # Complete the orthogonal basis using HPC
                    matrix_hpc = HPC16x8.from_array(u)
                    u = _convert_to_float32(matrix_hpc.complete_basis())
                
                return u, s, mx.transpose(v)
            return s
        else:
            # Similar process for m < n case
            aat = mx.matmul(a_array, mx.transpose(a_array))
            s = mx.sqrt(mx.abs(mx.linalg.eigvalsh(aat)))
            s = _to_sorted_array(s, reverse=True)[:k]
            
            if compute_uv:
                matrix_hpc = HPC16x8.from_array(aat)
                _, u = matrix_hpc.eig()
                u = _convert_to_float32(u)
                u = u[:, mx.argsort(-s)][:, :k]
                
                v = mx.zeros((n, k), dtype=a_array.dtype)
                for i in range(k):
                    if mx.greater(s[i], mx.array(1e-10)):
                        v_col = mx.divide(mx.matmul(mx.transpose(a_array), u[:, i]), s[i])
                        v = _update_array(v, slice(None), v_col.reshape(-1, 1))
                    else:
                        v_col = mx.zeros((n,), dtype=a_array.dtype)
                        v_col = _update_array(v_col, i % n, mx.array(1.0))
                        v = _update_array(v, slice(None), v_col.reshape(-1, 1))
                
                if full_matrices and n > k:
                    matrix_hpc = HPC16x8.from_array(v)
                    v = _convert_to_float32(matrix_hpc.complete_basis())
                
                return u, s, mx.transpose(v)
            return s

def eigvals(a: TensorLike) -> mx.array:
    """
    Compute the eigenvalues of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Eigenvalues of the matrix
    """
    # Convert input to MLX array with float32 dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_obj = Tensor.convert_to_tensor(a)
    eigenvalues, _ = eig(tensor_obj)
    return eigenvalues

def eig(a: TensorLike) -> Tuple[mx.array, mx.array]:
    """
    Compute eigenvalues and eigenvectors of a matrix.
    
    For symmetric matrices, uses MLX's native eigh. For non-symmetric or
    numerically sensitive cases, falls back to HPC implementation.
    
    Args:
        a: Input matrix (must be square)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    # Check if matrix is approximately symmetric
    diff = mx.subtract(a_array, mx.transpose(a_array))
    abs_diff = mx.abs(diff)
    is_symmetric = mx.all(mx.less(abs_diff, mx.array(1e-6))).item()
    
    try:
        if is_symmetric:
            # Use MLX's native eigh for symmetric matrices
            w, v = mx.linalg.eigh(a_array)
            # Sort in descending order
            sort_idx = mx.argsort(-w)
            w = w[sort_idx]
            v = v[:, sort_idx]
            return w, v
    except Exception as e:
        print(f"Native eigh failed: {e}, using HPC implementation")
    
    # Fall back to HPC implementation for non-symmetric or problematic cases
    matrix_hpc = HPC16x8.from_array(a_array)
    return matrix_hpc.eig()

def eigh(matrix: TensorLike) -> tuple[mx.array, mx.array]:
    """Compute eigenvalues and eigenvectors of a Hermitian/symmetric matrix.
    
    Args:
        matrix: Square matrix of shape (..., M, M) that is Hermitian/symmetric
    
    Returns:
        Tuple of:
            - eigenvalues (..., M) in ascending order
            - eigenvectors (..., M, M) where v[..., :, i] is eigenvector i
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    tensor = MLXTensor()
    
    matrix = tensor.convert_to_tensor(matrix)
    eigenvals, eigenvecs = mx.linalg.eigh(matrix)
    return eigenvals, eigenvecs

def _add_double_single(a_high, a_low, b_high, b_low):
    """Helper for double-single precision arithmetic."""
    s = a_high + b_high
    e = (a_high - s) + b_high + a_low + b_low
    return s, e

def _standard_qr(a_array: mx.array) -> Tuple[mx.array, mx.array]:
    """Standard QR decomposition for well-conditioned matrices."""
    try:
        # Try MLX's native QR first
        return mx.linalg.qr(a_array)
    except Exception as e:
        print(f"Native QR failed: {e}, using HPC implementation")
        # Fall back to HPC implementation
        matrix_hpc = HPC16x8.from_array(a_array)
        return _convert_hpc_result(matrix_hpc)

def _custom_qr(matrix_high: mx.array, matrix_low: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
    """MLX-specific QR decomposition with increased numerical stability."""
    rows, cols = matrix_high.shape
    
    # Use HPC implementation for non-square matrices
    if rows != cols:
        matrix_hpc = HPC16x8.from_array(matrix_high)
        q_hpc, r_hpc = matrix_hpc.qr()
        return _convert_to_float32(q_hpc), _convert_to_float32(r_hpc)
    
    # For square matrices, create HPC object with high/low parts
    if matrix_low is not None:
        matrix_hpc = HPC16x8(matrix_high, matrix_low)
    else:
        matrix_hpc = HPC16x8.from_array(matrix_high)
    
    return _convert_hpc_result(matrix_hpc)

def qr(a: TensorLike, mode: Literal["reduced", "complete", "r", "raw"] = "reduced") -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    Compute QR decomposition of a matrix.
    
    Args:
        a: Input matrix
        mode: Type of decomposition to compute
            - 'reduced': Return q, r with shapes (M,K), (K,N) where K = min(M,N)
            - 'complete': Return q, r with shapes (M,M), (M,N)
            - 'r': Return only r
            - 'raw': Return only q
            
    Returns:
        - mode='reduced': q, r arrays
        - mode='complete': q, r arrays
        - mode='r': r array only
        - mode='raw': q array only
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    
    # Get QR decomposition
    q, r = _custom_qr(a_array)
    
    # Handle different modes
    if mode == "reduced":
        m, n = a_array.shape
        k = min(m, n)
        return q[:, :k], r[:k]
    elif mode == "complete":
        if q.shape[0] > q.shape[1]:
            # Need to complete the basis
            matrix_hpc = HPC16x8.from_array(q)
            q = matrix_hpc.complete_basis().to_float32()
        return q, r
    elif mode == "r":
        return r
    else:  # mode == "raw"
        return q

def _eigsh_power_iteration(a: mx.array, k: int = 6, num_iterations: int = 100) -> Tuple[mx.array, mx.array]:
    """
    Compute largest k eigenvalues/vectors using power iteration.
    
    Uses MLX operations throughout for GPU acceleration.
    
    Args:
        a: Input symmetric matrix
        k: Number of eigenvalues to compute
        num_iterations: Maximum power iterations
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    n = a.shape[0]
    v = mx.random.normal((n, k))
    
    # Orthogonalize initial vectors
    q = mx.zeros((n, k), dtype=a.dtype)
    for j in range(k):
        vj = v[:, j]
        for i in range(j):
            vj = mx.subtract(vj, mx.multiply(mx.sum(mx.multiply(q[:, i], vj)), q[:, i]))
        vj_norm = mx.linalg.norm(vj)
        if mx.greater(vj_norm, mx.array(1e-10)):
            q = _update_array(q, slice(None), mx.divide(vj, vj_norm))
        else:
            q = _update_array(q, slice(None), mx.zeros_like(vj))
    
    # Power iteration
    for _ in range(num_iterations):
        # Update vectors
        v = mx.matmul(a, q)
        
        # Orthogonalize
        for j in range(k):
            vj = v[:, j]
            for i in range(j):
                vj = mx.subtract(vj, mx.multiply(mx.sum(mx.multiply(q[:, i], vj)), q[:, i]))
            vj_norm = mx.linalg.norm(vj)
            if mx.greater(vj_norm, mx.array(1e-10)):
                q = _update_array(q, slice(None), mx.divide(vj, vj_norm))
    
    # Compute Rayleigh quotients
    eigenvalues = mx.zeros(k, dtype=a.dtype)
    for i in range(k):
        eigenvalues = _update_array(eigenvalues, i, mx.sum(mx.multiply(q[:, i], mx.matmul(a, q[:, i]))))
    
    return eigenvalues, q

def _complete_orthogonal_basis(u: mx.array, full_size: Optional[int] = None) -> mx.array:
    """Complete orthogonal basis using Metal acceleration when available."""
    try:
        # Try using Metal-accelerated implementation
        from ember_ml.backend.mlx.linearalg.hpc_nonsquare import complete_orthogonal_basis_metal
        return complete_orthogonal_basis_metal(u)
    except Exception as e:
        print(f"Metal acceleration failed: {e}, using standard implementation")
        if full_size is None:
            full_size = u.shape[0]
        
        # Fall back to standard implementation
        m = u.shape[0]
        current_size = u.shape[1]
        remaining = full_size - current_size
        
        if remaining <= 0:
            return u
            
        # Generate random vectors and orthogonalize
        additional = mx.random.normal((m, remaining))
        result = mx.concatenate([u, additional], axis=1)
        
        # Use block Gram-Schmidt for better numerical stability
        block_size = 32
        for start in range(0, remaining, block_size):
            end = min(start + block_size, remaining)
            current_block = result[:, current_size + start:current_size + end]
            
            # Orthogonalize against previous vectors
            for i in range(current_size + start):
                proj = mx.sum(mx.multiply(result[:, i:i+1], current_block), axis=0)
                current_block = mx.subtract(
                    current_block,
                    mx.multiply(result[:, i:i+1], proj.reshape(1, -1))
                )
            
            # Normalize columns
            norms = mx.sqrt(mx.sum(mx.square(current_block), axis=0))
            nonzero = mx.greater(norms, mx.array(1e-10))
            safe_norms = mx.where(nonzero, norms, mx.ones_like(norms))
            current_block = mx.divide(current_block, safe_norms.reshape(1, -1))
            
            # Replace close-to-zero columns with new random vectors
            for j in range(end - start):
                if not nonzero[j].item():
                    new_vec = mx.random.normal((m,))
                    current_block = _update_array(current_block, slice(None), new_vec)
            
            # Update result
            result = _update_array(result, slice(None), current_block)
        
        return result
