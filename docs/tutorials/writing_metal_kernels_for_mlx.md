# Writing Metal Kernels for MLX: A Comprehensive Guide

## Introduction

MLX provides a high-level interface for machine learning on Apple Silicon, but sometimes you need to write custom Metal kernels for performance-critical operations. This guide covers the process of writing, debugging, and optimizing Metal kernels for MLX, with a focus on practical examples and common pitfalls.

## Basic Structure of a Metal Kernel in MLX

### Kernel Definition

```python
import mlx.core as mx

# Define the kernel source code as a string
kernel_source = """
// Your Metal kernel code here
"""

# Compile the kernel
compiled_kernel = mx.fast.metal_kernel(
    name="your_kernel_name",
    source=kernel_source,
    input_names=["input1", "input2"],
    output_names=["output"],
    ensure_row_contiguous=True
)

# Use the kernel
def your_function(input1, input2):
    outputs = compiled_kernel(
        inputs=[input1, input2],
        output_shapes=[(output_shape)],
        output_dtypes=[mx.float32],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1)
    )
    return outputs[0]
```

## Metal Kernel Syntax and Best Practices

### Thread Identification

| Component | Description | Best Practice |
|-----------|-------------|--------------|
| `thread_position_in_grid` | Position of the thread in the entire grid | Access the `.x` component for 1D indexing: `uint tid = thread_position_in_grid.x;` |
| `threads_per_threadgroup` | Number of threads in a threadgroup | Access the `.x` component: `uint num_threads = threads_per_threadgroup.x;` |
| `thread_position_in_threadgroup` | Position within the current threadgroup | Useful for shared memory operations |
| `threadgroups_per_grid` | Number of threadgroups in the grid | Rarely needed directly |

### Example: Basic Thread Indexing

```metal
uint tid = thread_position_in_grid.x;
uint num_threads = threads_per_threadgroup.x;

// Process elements in parallel with proper striding
for (uint idx = tid; idx < n; idx += num_threads) {
    output[idx] = input[idx] * 2.0f;
}
```

### SIMD Group Operations

For efficient parallel reductions, use SIMD group operations:

```metal
uint simd_lane_id = tid % 32;  // 32 is the WARP_SIZE
uint simd_group_id = tid / 32;

// Each thread computes its partial result
float thread_sum = 0.0f;
for (uint i = tid; i < n; i += num_threads) {
    thread_sum += input[i];
}

// Reduce within SIMD group (much faster than manual reduction)
thread_sum = simd_sum(thread_sum);
```

## Shared Memory Management

### Declaration and Size Limits

| Aspect | Details | Pitfalls |
|--------|---------|----------|
| Declaration | `threadgroup float shared_mem[SIZE];` | Must be declared at kernel scope |
| Size Limit | 32KB (32,768 bytes) total per threadgroup | Exceeding this will cause compilation failure |
| Data Types | Each `float` is 4 bytes | A 1000-element float array uses 4000 bytes |
| Dynamic Sizing | Not directly supported | Use constants or preprocessor defines |

### Example: Calculating Shared Memory Size

```metal
// For a float array of 1000 elements:
// 1000 * 4 bytes = 4000 bytes (well under the 32KB limit)
threadgroup float shared_mem[1000];

// For a 2D matrix of 100x100 floats:
// 100 * 100 * 4 bytes = 40,000 bytes (exceeds the 32KB limit!)
// Instead, use a smaller size:
threadgroup float shared_matrix[80 * 80]; // 25,600 bytes (under the limit)
```

### Synchronization

Always use barriers when accessing shared memory:

```metal
// Write to shared memory
shared_mem[tid] = input[tid];

// Ensure all threads have written to shared memory
threadgroup_barrier(mem_flags::mem_device);

// Now read from shared memory
float value = shared_mem[other_idx];
```

## Common Patterns for Matrix Operations

### Element-wise Operations

```metal
uint tid = thread_position_in_grid.x;
uint n = shape[0];
uint num_threads = threads_per_threadgroup.x;

// Process elements in parallel
for (uint idx = tid; idx < n; idx += num_threads) {
    output[idx] = func(input[idx]);
}
```

### Matrix Multiplication

```metal
uint tid = thread_position_in_grid.x;
uint m = shape[0];
uint n = shape[1];
uint k = shape[2];
uint num_threads = threads_per_threadgroup.x;

// Each thread computes one or more elements of the output
for (uint idx = tid; idx < m * n; idx += num_threads) {
    uint row = idx / n;
    uint col = idx % n;
    
    float sum = 0.0f;
    for (uint i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + col];
    }
    
    C[row * n + col] = sum;
}
```

### Reduction Operations

```metal
uint tid = thread_position_in_grid.x;
uint n = shape[0];
uint num_threads = threads_per_threadgroup.x;
uint simd_lane_id = tid % 32;
uint simd_group_id = tid / 32;

// Each thread computes partial sum
float thread_sum = 0.0f;
for (uint i = tid; i < n; i += num_threads) {
    thread_sum += input[i];
}

// Reduce within SIMD group
thread_sum = simd_sum(thread_sum);

// First thread in each SIMD group writes to shared memory
threadgroup float shared_sums[8];  // Assuming max 8 SIMD groups
if (simd_lane_id == 0 && simd_group_id < 8) {
    shared_sums[simd_group_id] = thread_sum;
}

threadgroup_barrier(mem_flags::mem_device);

// Thread 0 combines results
if (tid == 0) {
    float total_sum = 0.0f;
    for (uint i = 0; i < min(8u, (num_threads + 31) / 32); i++) {
        total_sum += shared_sums[i];
    }
    output[0] = total_sum;
}
```

## Advanced Techniques

### Tiling for Matrix Operations

For large matrices, use tiling to improve cache efficiency:

```metal
#define TILE_SIZE 16

uint tid = thread_position_in_grid.x;
uint row = tid / TILE_SIZE;
uint col = tid % TILE_SIZE;

threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

float sum = 0.0f;
for (uint t = 0; t < k; t += TILE_SIZE) {
    // Load tiles into shared memory
    if (row < m && t + col < k)
        A_tile[row][col] = A[row * k + (t + col)];
    else
        A_tile[row][col] = 0.0f;
    
    if (t + row < k && col < n)
        B_tile[row][col] = B[(t + row) * n + col];
    else
        B_tile[row][col] = 0.0f;
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Compute partial sum for this tile
    for (uint i = 0; i < TILE_SIZE; i++)
        sum += A_tile[row][i] * B_tile[i][col];
    
    threadgroup_barrier(mem_flags::mem_device);
}

if (row < m && col < n)
    C[row * n + col] = sum;
```

### Coalesced Memory Access

Ensure threads access memory in a coalesced pattern for better performance:

```metal
// Good: Consecutive threads access consecutive memory locations
for (uint idx = tid; idx < n; idx += num_threads) {
    output[idx] = input[idx];
}

// Bad: Consecutive threads access memory with large strides
for (uint i = 0; i < n / num_threads; i++) {
    output[tid + i * num_threads] = input[tid + i * num_threads];
}
```

## Common Pitfalls and Solutions

### 1. Exceeding Shared Memory Limits

**Problem**: Metal has a 32KB limit on shared memory per threadgroup.

**Solution**: 
- Calculate your shared memory usage carefully: `num_elements * sizeof(element_type)`
- For large matrices, process them in smaller tiles
- Reduce the maximum dimensions you support (e.g., limit MAX_K to 32 instead of 512)

**Example Fix**:
```metal
// Before (exceeds limit for large matrices)
threadgroup float shared_Z[4096 * 64];  // 1MB for 4096x64 matrix!

// After (stays under 32KB limit)
threadgroup float shared_Z[250 * 32];  // ~32KB for 250x32 matrix
```

### 2. Incorrect Thread ID Access

**Problem**: Using `thread_position_in_grid` directly instead of accessing its components.

**Solution**: Always access the appropriate component (usually `.x` for 1D indexing).

**Example Fix**:
```metal
// Before (incorrect)
uint tid = thread_position_in_grid;  // Error: cannot convert uint3 to uint

// After (correct)
uint tid = thread_position_in_grid.x;  // Access the x component
```

### 3. Missing Synchronization Barriers

**Problem**: Reading shared memory before all threads have written to it.

**Solution**: Always use `threadgroup_barrier(mem_flags::mem_device)` after writing to shared memory.

**Example Fix**:
```metal
// Before (race condition)
shared_mem[tid] = input[tid];
float value = shared_mem[other_idx];  // May read uninitialized data

// After (correct)
shared_mem[tid] = input[tid];
threadgroup_barrier(mem_flags::mem_device);
float value = shared_mem[other_idx];  // Safe to read now
```

### 4. Inefficient Reduction

**Problem**: Manual reduction is slow and complex.

**Solution**: Use SIMD operations like `simd_sum` for efficient reduction.

**Example Fix**:
```metal
// Before (manual reduction)
if (tid < 32) {
    for (uint s = 1; s < 32; s *= 2) {
        float other = shared_mem[tid ^ s];
        shared_mem[tid] += other;
    }
}

// After (using SIMD operations)
float thread_sum = /* your value */;
thread_sum = simd_sum(thread_sum);
```

### 5. Incorrect Function Declarations

**Problem**: Including function declarations in the kernel source causes compilation errors.

**Solution**: Let MLX handle the function declaration; only provide the function body.

**Example Fix**:
```metal
// Before (incorrect)
kernel void your_kernel(
    const device float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    uint thread_position_in_grid [[thread_position_in_grid]])
{
    // Kernel code
}

// After (correct)
// No function declaration, just the body
uint tid = thread_position_in_grid.x;
// Kernel code
```

### 6. Thread Divergence

**Problem**: Different execution paths within a SIMD group cause performance issues.

**Solution**: Minimize conditional code within the same SIMD group.

**Example Fix**:
```metal
// Before (divergent)
if (tid % 2 == 0) {
    // Path A
} else {
    // Path B
}

// After (less divergent)
// Group similar operations together
for (uint idx = tid; idx < n; idx += num_threads) {
    if (idx % 2 == 0) {
        // All threads process even indices
    }
}
for (uint idx = tid; idx < n; idx += num_threads) {
    if (idx % 2 == 1) {
        // All threads process odd indices
    }
}
```

## Debugging Strategies

### 1. Incremental Testing

Start with a minimal kernel and gradually add complexity:

1. Begin with a simple element-wise operation
2. Add shared memory usage
3. Add SIMD operations
4. Add complex algorithms

This helps isolate where issues occur.

### 2. Controlled Inputs

Use inputs with known expected outputs:

```python
# Create a matrix with known singular values for SVD testing
U, _ = qr(mx.random.normal((n, n)))
s_values = mx.linspace(n, 1, n)
V, _ = qr(mx.random.normal((n, n)))

# Create diagonal matrix
s_diag = mx.zeros((n, n))
for i in range(n):
    s_diag = scatter(mx.array([i, i]), s_values[i], s_diag.shape)

# Create test matrix
A = mx.matmul(U, mx.matmul(s_diag, mx.transpose(V)))
```

### 3. Error Message Analysis

Common error messages and their meanings:

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `Threadgroup memory size exceeds the maximum` | Too much shared memory | Reduce array sizes |
| `Unable to build metal library from source` | Syntax error in kernel | Check for typos, missing braces |
| `Cannot initialize a variable of type 'uint' with an lvalue of type 'uint3'` | Incorrect thread ID access | Use `.x` component |
| `Extraneous closing brace` | Incorrect function structure | Remove function declaration |

### 4. Sandbox Testing

Create isolated test files to debug specific issues:

```python
def test_simple_kernel():
    kernel_source = """
    // Simple kernel for testing
    uint tid = thread_position_in_grid.x;
    output[tid] = input[tid];
    """
    
    kernel = mx.fast.metal_kernel(
        name="test_kernel",
        source=kernel_source,
        input_names=["input"],
        output_names=["output"],
        ensure_row_contiguous=True
    )
    
    # Test with simple data
    input_data = mx.ones((32,))
    outputs = kernel(
        inputs=[input_data],
        output_shapes=[(32,)],
        output_dtypes=[mx.float32],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1)
    )
    
    # Verify result
    print(outputs[0])
```

## Performance Optimization

### 1. Thread Configuration

Choose appropriate grid and threadgroup sizes:

```python
# For small operations (< 1024 elements)
grid = (32, 1, 1)
threadgroup = (32, 1, 1)

# For medium operations
grid = (256, 1, 1)
threadgroup = (256, 1, 1)

# For large operations
grid = (1024, 1, 1)
threadgroup = (256, 1, 1)  # Keep threadgroup size reasonable
```

### 2. Memory Access Patterns

Optimize memory access for better performance:

- Coalesced access: Adjacent threads access adjacent memory
- Minimize bank conflicts in shared memory
- Use appropriate data types (float32 vs float16)

### 3. Algorithmic Optimizations

- Use tiling for matrix operations
- Leverage SIMD operations for reductions
- Balance work across threads evenly

## Real-World Example: SVD Power Iteration

Here's a complete example of a power iteration kernel for SVD:

```python
import mlx.core as mx

# Define the kernel source
kernel_source = """
#define EPSILON 1e-10f
#define MAX_K 32
#define WARP_SIZE 32

uint tid = thread_position_in_grid.x;
uint num_threads = threads_per_threadgroup.x;
uint simd_lane_id = tid % WARP_SIZE;
uint simd_group_id = tid / WARP_SIZE;

uint n = shapeParams[0];
uint k = shapeParams[1];
uint num_iterations = iterParams[0];
float tolerance = tolParams[0];

// Shared memory (carefully sized to stay under 32KB limit)
threadgroup float shared_Z[250 * MAX_K];
threadgroup float shared_proj[MAX_K];
threadgroup float shared_norm[MAX_K];

// Initialize Q_out with Q_init
for (uint idx = tid; idx < n * k; idx += num_threads) {
    Q_out[idx] = Q_init[idx];
}

threadgroup_barrier(mem_flags::mem_device);

// Power iteration with Gram-Schmidt orthogonalization
for (uint iter = 0; iter < num_iterations; iter++) {
    // Matrix multiplication: Z = A * Q_out
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        uint row = idx / k;
        uint col = idx % k;
        
        float sum = 0.0f;
        for (uint i = 0; i < n; i++) {
            sum += A[row * n + i] * Q_out[i * k + col];
        }
        shared_Z[idx] = sum;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Gram-Schmidt orthogonalization
    for (uint col = 0; col < k; col++) {
        // Orthogonalize against previous columns
        for (uint j = 0; j < col; j++) {
            // Compute dot product
            float thread_proj = 0.0f;
            for (uint row = tid; row < n; row += num_threads) {
                thread_proj += Q_out[row * k + j] * shared_Z[row * k + col];
            }
            
            // Reduce using SIMD operations
            thread_proj = simd_sum(thread_proj);
            
            // First thread in each SIMD group writes to shared memory
            if (simd_lane_id == 0 && simd_group_id < 8) {
                shared_proj[simd_group_id] = thread_proj;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 combines results
            if (tid == 0) {
                float proj = 0.0f;
                for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                    proj += shared_proj[i];
                }
                shared_proj[0] = proj;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            float proj = shared_proj[0];
            
            // Subtract projection
            for (uint row = tid; row < n; row += num_threads) {
                shared_Z[row * k + col] -= proj * Q_out[row * k + j];
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
        
        // Compute norm
        float thread_norm_sq = 0.0f;
        for (uint row = tid; row < n; row += num_threads) {
            float val = shared_Z[row * k + col];
            thread_norm_sq += val * val;
        }
        
        // Reduce using SIMD operations
        thread_norm_sq = simd_sum(thread_norm_sq);
        
        // First thread in each SIMD group writes to shared memory
        if (simd_lane_id == 0 && simd_group_id < 8) {
            shared_norm[simd_group_id] = thread_norm_sq;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Thread 0 computes final norm
        if (tid == 0) {
            float norm_sq = 0.0f;
            for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                norm_sq += shared_norm[i];
            }
            float norm = sqrt(norm_sq);
            shared_norm[0] = norm;
            shared_norm[1] = (norm > tolerance) ? (1.0f / norm) : 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        float norm = shared_norm[0];
        float inv_norm = shared_norm[1];
        
        // Normalize column
        for (uint row = tid; row < n; row += num_threads) {
            if (norm > tolerance) {
                Q_out[row * k + col] = shared_Z[row * k + col] * inv_norm;
            } else {
                Q_out[row * k + col] = 0.0f;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
    }
}
"""

# Compile the kernel
power_iter_kernel = mx.fast.metal_kernel(
    name="power_iter_kernel",
    source=kernel_source,
    input_names=["A", "Q_init", "shapeParams", "iterParams", "tolParams"],
    output_names=["Q_out"],
    ensure_row_contiguous=True
)

# Function to call the kernel
def power_iteration(A, Q_init, num_iterations=10, tolerance=1e-10):
    n, k = Q_init.shape
    shape_params = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([num_iterations], dtype=mx.uint32)
    tol_params = mx.array([tolerance], dtype=mx.float32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Call the kernel
    outputs = power_iter_kernel(
        inputs=[A, Q_init, shape_params, iter_params, tol_params],
        output_shapes=[(n, k)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup
    )
    
    return outputs[0]
```

## Conclusion

Writing efficient Metal kernels for MLX requires understanding both the Metal Shading Language and MLX's kernel compilation system. By following the best practices and avoiding common pitfalls outlined in this guide, you can create high-performance custom operations that leverage the full power of Apple Silicon.

Remember to:
1. Start simple and add complexity incrementally
2. Carefully manage shared memory usage
3. Use appropriate synchronization barriers
4. Leverage SIMD operations for efficient reductions
5. Test thoroughly with controlled inputs

With these principles in mind, you'll be able to write robust and efficient Metal kernels for your MLX applications.