# Backend Normalization Plan

## Overview

This document outlines a plan to normalize operations across all backends (NumPy, PyTorch, MLX) in the EmberHarmony framework. The goal is to ensure that all operations are available and behave consistently across all backends, making the codebase truly backend-agnostic.

## Current Status

We've successfully implemented:
- Detection of Python operators that break lazy evaluation
- Guidelines for using ops functions instead of Python operators
- Tests for the detection tool and benchmark script

## Next Steps: Backend Normalization

### 1. Identify Missing Operations

First, we need to identify operations that are implemented in some backends but not others:

- **Create an operation inventory**: List all operations implemented in each backend
- **Identify gaps**: Find operations that are missing in one or more backends
- **Prioritize operations**: Focus on high-impact operations used frequently in the codebase

### 2. Implement Missing Operations

For each missing operation:

- **Research equivalent implementations**: Find ways to implement the operation using existing primitives
- **Implement the operation**: Add the missing implementation to the appropriate backend
- **Test for consistency**: Ensure the operation behaves the same across all backends

### 3. Ensure Consistent Behavior

Even for operations that exist across all backends, we need to ensure they behave consistently:

- **Define expected behavior**: Document the expected behavior for each operation
- **Create test cases**: Develop tests that verify consistent behavior across backends
- **Fix inconsistencies**: Modify implementations to ensure consistent behavior

### 4. Special Focus Areas

#### 4.1 Random Operations

Random operations are particularly challenging to normalize because:
- Different backends use different random number generators
- Seeding behavior may differ
- Distribution parameters may be interpreted differently

We need to:
- Ensure consistent seeding behavior
- Verify that distributions have the same statistical properties
- Create tests that compare distribution outputs

#### 4.2 Device Placement

Device placement varies significantly across backends:
- NumPy is CPU-only
- PyTorch supports CPU, CUDA, MPS
- MLX supports CPU, Metal

We need to:
- Create a consistent device abstraction
- Implement fallbacks for unsupported devices
- Test device placement behavior

#### 4.3 Data Types

Data type handling differs across backends:
- Different naming conventions
- Different supported types
- Different default types

We need to:
- Create a consistent type system
- Implement type conversion utilities
- Test type handling behavior

## Implementation Plan

### Phase 1: Inventory and Gap Analysis

1. Create a complete inventory of operations in each backend
2. Identify gaps and inconsistencies
3. Prioritize operations for implementation

### Phase 2: Implementation

1. Implement high-priority missing operations
2. Create tests for each implementation
3. Document implementation details

### Phase 3: Validation

1. Run comprehensive tests across all backends
2. Verify consistent behavior
3. Benchmark performance

### Phase 4: Documentation and Guidelines

1. Update documentation with new operations
2. Create guidelines for implementing new operations
3. Document any remaining limitations or differences

## Success Criteria

- All core operations are available across all backends
- Operations behave consistently across backends
- Comprehensive tests verify consistency
- Documentation covers all operations and their behavior