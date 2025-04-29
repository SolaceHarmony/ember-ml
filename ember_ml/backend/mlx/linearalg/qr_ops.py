# Content for ember_ml/backend/mlx/linearalg/qr_ops.py (Device Memory for Q_temp)
from __future__ import annotations
"""
MLX QR decomposition operations for ember_ml.

This module provides MLX implementations of QR decomposition using
a custom Metal kernel for GPU acceleration.
"""

from typing import Union, Tuple, Literal, Optional
import mlx.core as mx

# Import from tensor_ops and types
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.tensor import MLXTensor

dtype_obj = MLXDType()


# ============================================================================
#  enhanced_hpc_qr.py   •   drop-in, Metal + MLX
# ============================================================================


import time
from typing import Tuple, Optional
import mlx.core as mx

# ----------------------------------------------------------------------------
# 1 · Metal kernel                                                            #
# ----------------------------------------------------------------------------
_ENHANCED_QR_SRC = r"""

/* ------------------------------------------------------------------ constants */
#define EPSILON     1e-10f
#define NUM_LIMBS   8u          // 128-bit accumulator (8 × 16-bit)
#define LIMB_RADIX  65536.0f    // 2¹⁶
#define WARP_SIZE   32u         // Threads per warp/wavefront
    uint                tid;     // Thread ID in grid
    uint3               tpg;     // Threadgroup size
    uint3               gpg;     // Grid size
    uint                simd_lane_id = tid % WARP_SIZE;  // Lane ID within SIMD group
    uint                simd_group_id = tid / WARP_SIZE; // SIMD group ID

    const uint m        = shape[0];
    const uint n        = shape[1];
    const uint min_dim  = (m < n ? m : n);
    const uint grid_sz  = gpg.x * tpg.x;    // Total number of threads in grid

    /* 0 · initialise Q ← I,  R ← A (parallel over full grid) */
    for (uint idx = tid; idx < m * m; idx += grid_sz) {
        uint r = idx / m, c = idx % m;
        Q_out[idx] = (r == c ? 1.0f : 0.0f);
    }
    for (uint idx = tid; idx < m * n; idx += grid_sz)
        R_out[idx] = A[idx];

    threadgroup_barrier(mem_flags::mem_device);

    /* ===== parallel QR decomposition ===== */
    for (uint k = 0; k < min_dim; ++k)
    {
        /* -- column scaling (improves robustness) ------------------ */
        // Each thread finds max in its assigned range
        float thread_max = 0.0f;
        for (uint i = k + tid; i < m; i += grid_sz) {
            thread_max = fmax(thread_max, fabs(R_out[i*n + k]));
        }
        
        // Reduce max across threads in same SIMD group
        thread_max = simd_max(thread_max);
        
        // First thread in each SIMD group writes to shared memory
        threadgroup float simd_max[8]; // Assuming max 8 SIMD groups
        if (simd_lane_id == 0 && simd_group_id < 8) {
            simd_max[simd_group_id] = thread_max;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // First thread finds max across SIMD groups
        float cmax = 0.0f;
        if (tid == 0) {
            for (uint i = 0; i < min(8u, (grid_sz + WARP_SIZE - 1) / WARP_SIZE); ++i) {
                cmax = fmax(cmax, simd_max[i]);
            }
            dbg[10] = cmax;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Broadcast cmax to all threads
        if (tid == 0) {
            simd_max[0] = cmax;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        cmax = simd_max[0];
        float scale = (cmax > EPSILON ? 1.0f / cmax : 1.0f);
        
        // Scale column k in parallel
        for (uint i = k + tid; i < m; i += grid_sz) {
            R_out[i*n + k] *= scale;
        }
        
        threadgroup_barrier(mem_flags::mem_device);

        /* -- build Householder v ----------------------------------- */
        // Each thread computes partial sum for sigma
        float partial_sigma = 0.0f;
        for (uint i = k + tid; i < m; i += grid_sz) {
            float v = R_out[i*n + k];
            partial_sigma += v*v;
        }
        
        // Reduce sigma across threads
        partial_sigma = simd_sum(partial_sigma);
        
        // First thread in each SIMD group writes to shared memory
        threadgroup float simd_sigma[8];
        if (simd_lane_id == 0 && simd_group_id < 8) {
            simd_sigma[simd_group_id] = partial_sigma;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // First thread combines results
        float sigma = 0.0f;
        float norm = 0.0f;
        float sign = 0.0f;
        if (tid == 0) {
            for (uint i = 0; i < min(8u, (grid_sz + WARP_SIZE - 1) / WARP_SIZE); ++i) {
                sigma += simd_sigma[i];
            }
            dbg[4] = sigma;
            norm = sqrt(sigma);
            dbg[5] = norm;
            
            // Check if norm is too small
            if (norm < EPSILON) {
                // Unscale the diagonal element
                R_out[k*n + k] /= scale;
                simd_sigma[0] = -1.0f; // Signal to skip this iteration
            } else {
                sign = (R_out[k*n + k] >= 0.0f ? 1.0f : -1.0f);
                R_out[k*n + k] += sign * norm;  // v₀ update
                simd_sigma[0] = 0.0f; // Signal to continue
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Check if we should skip this iteration
        if (simd_sigma[0] < 0.0f) {
            continue;
        }
        
        threadgroup_barrier(mem_flags::mem_device);

        /* -- limb-precision vᵀv (parallel version) ----------------- */
        // Each thread processes its assigned elements
        threadgroup uint thread_limbs[WARP_SIZE * NUM_LIMBS];
        
        // Initialize thread_limbs to zero
        for (uint l = tid; l < WARP_SIZE * NUM_LIMBS; l += grid_sz) {
            thread_limbs[l] = 0u;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Each thread computes partial limbs
        uint local_limb[NUM_LIMBS] = {0u};
        for (uint i = k + tid; i < m; i += grid_sz) {
            uint bits = as_type<uint>(R_out[i*n + k]);
            ushort lo = bits & 0xFFFFu;
            ushort hi = (bits >> 16) & 0xFFFFu;
            uint p0 = uint(lo*lo);
            uint p1 = uint(hi*hi);
            uint pc = uint(lo*hi) << 1;

            local_limb[0] +=  p0 & 0xFFFFu;
            local_limb[1] += (p0 >> 16) + (pc & 0xFFFFu);
            local_limb[2] += (pc >> 16) + (p1 & 0xFFFFu);
            local_limb[3] +=  p1 >> 16;
        }
        
        // Store local limbs to shared memory
        for (uint l = 0; l < NUM_LIMBS; ++l) {
            thread_limbs[tid * NUM_LIMBS + l] = local_limb[l];
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // First thread combines all limbs
        uint combined_limb[NUM_LIMBS] = {0u};
        float vtv = 0.0f;
        float inv_vtv = 0.0f;
        
        if (tid == 0) {
            // Combine all thread limbs
            for (uint t = 0; t < grid_sz; ++t) {
                for (uint l = 0; l < NUM_LIMBS; ++l) {
                    combined_limb[l] += thread_limbs[t * NUM_LIMBS + l];
                }
            }
            
            // Carry propagation
            for (uint l = 0; l < NUM_LIMBS-1; ++l) {
                uint carry = combined_limb[l] >> 16;
                combined_limb[l] &= 0xFFFFu;
                combined_limb[l+1] += carry;
            }
            
            // Convert to float
            float radix = 1.0f;
            for (uint l = 0; l < NUM_LIMBS; ++l) {
                vtv += float(combined_limb[l]) * radix;
                radix *= LIMB_RADIX;
            }
            
            dbg[6] = vtv;
            inv_vtv = (vtv > EPSILON ? 1.0f / vtv : 0.0f);
            dbg[7] = inv_vtv;
            
            // Store for other threads to access
            thread_limbs[0] = (inv_vtv == 0.0f ? 1u : 0u); // Flag for skipping
            
            // Store vtv and inv_vtv for other threads
            simd_sigma[1] = vtv;
            simd_sigma[2] = inv_vtv;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Check if we should skip reflection
        if (thread_limbs[0] == 1u) {
            // Unscale the column in parallel
            for (uint i = k + tid; i < m; i += grid_sz) {
                R_out[i*n + k] /= scale;
            }
            continue;
        }
        
        // Get shared values
        vtv = simd_sigma[1];
        inv_vtv = simd_sigma[2];
        
        threadgroup_barrier(mem_flags::mem_device);
        
        /* -- reflect R (k … n-1) in parallel ----------------------- */
        // Each thread handles a subset of columns
        for (uint j = k + tid; j < n; j += grid_sz) {
            // Calculate dot product for this column
            float dot = 0.0f;
            for (uint i = k; i < m; ++i) {
                dot += R_out[i*n + k] * R_out[i*n + j];
            }
            
            // Store debug info for first column
            if (j == k && tid == 0) {
                dbg[8] = dot;
            }
            
            float beta = 2.0f * dot * inv_vtv;
            
            // Update column j
            for (uint i = k; i < m; ++i) {
                R_out[i*n + j] -= beta * R_out[i*n + k];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        /* -- reflect Q (0 … m-1) in parallel ----------------------- */
        // Each thread handles a subset of columns
        for (uint j = tid; j < m; j += grid_sz) {
            // Calculate dot product for this column
            float dot = 0.0f;
            for (uint i = k; i < m; ++i) {
                dot += R_out[i*n + k] * Q_out[i*m + j];
            }
            
            // Store debug info for first column
            if (j == 0 && tid == 0) {
                dbg[9] = dot;
            }
            
            float beta = 2.0f * dot * inv_vtv;
            
            // Update column j
            for (uint i = k; i < m; ++i) {
                Q_out[i*m + j] -= beta * R_out[i*n + k];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        /* -- un-scale column k in parallel ------------------------- */
        for (uint i = k + tid; i < m; i += grid_sz) {
            R_out[i*n + k] /= scale;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
    }

    /* -- force R upper-triangular (in parallel) -------------------- */
    for (uint r = 1 + tid; r < m; r += grid_sz) {
        for (uint c = 0; c < min(r, n); ++c) {
            R_out[r*n + c] = 0.0f;
        }
    }
    
    if (tid == 0) {
        dbg[15] = 1.0f;     // success flag
    }
"""

# ----------------------------------------------------------------------------
# 2 · compile the kernel                                                      #
# ----------------------------------------------------------------------------
_ENHANCED_QR_KERNEL = mx.fast.metal_kernel(
    name              = "enhanced_hpc_qr_kernel",
    source            = _ENHANCED_QR_SRC,
    input_names       = ["A", "shape"],
    output_names      = ["Q_out", "R_out", "dbg"],
    ensure_row_contiguous=True
)

# ----------------------------------------------------------------------------
# 3 · qr function                                                             #
# ----------------------------------------------------------------------------

def qr(A,
       *,
       debug: bool = False
      ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """
    Numerically-stable QR with limb accumulation using Metal GPU acceleration.
    
    This implementation uses a highly parallelized Metal kernel for maximum performance
    on Apple GPUs.

    Returns (Q, R[, dbg]).
    """
    A = mx.array(A, dtype=mx.float32)
    
    # Get dimensions
    m, n = A.shape
    
    # Prepare inputs
    shape = mx.array([m, n], dtype=mx.uint32)
    
    # Prepare outputs
    Q_out = mx.zeros((m, m), dtype=mx.float32)
    R_out = mx.zeros((m, n), dtype=mx.float32)
    dbg = mx.zeros(16, dtype=mx.float32)
    
    # Configure kernel execution
    grid = (min(1024, m * n), 1, 1)  # Adjust grid size based on matrix dimensions
    threadgroup = (min(256, grid[0]), 1, 1)  # Adjust threadgroup size for optimal performance
    
    # Execute the Metal kernel
    # Define output shapes and dtypes
    output_shapes = [(m, m), (m, n), (16,)]
    output_dtypes = [mx.float32, mx.float32, mx.float32]
    
    # Call the kernel with proper parameters according to MLX API
    outputs = _ENHANCED_QR_KERNEL(
        inputs=[A, shape],
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup
    )
    
    Q, R, dbg_out = outputs
    
    # Return results
    return (Q, R, dbg_out) if debug else (Q, R)

# ----------------------------------------------------------------------------
# 4 · quick self-test                                                         #
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Enhanced HPC-QR (fixed build)")
    print("="*72)
    for (m, n) in [(10, 10), (100, 150), (300, 300)]:
        A   = mx.random.normal((m, n))
        t0  = time.time()
        Q, R = qr(A)
        dt  = time.time() - t0

        ortho = mx.mean(mx.abs(mx.matmul(Q.T, Q) - mx.eye(m))).item()
        recon = mx.mean(mx.abs(mx.matmul(Q, R) - A)).item()
        print(f"{m:4d}×{n:<4d}  ‖QᵀQ−I‖₁={ortho:9.2e}   "
              f"‖QR−A‖₁={recon:9.2e}   {dt:6.3f}s")
    print("done ✓")