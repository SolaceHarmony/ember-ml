"""MLX backend matrix exponential using scaling-and-squaring + Pade-13."""

from __future__ import annotations

import math

from ember_ml import ops, tensor

__all__ = ["expm"]

_PADE13_COEFFS = (
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
)

_THETA_13 = 4.25


def expm(matrix: tensor.EmberTensor) -> tensor.EmberTensor:
    """Compute the matrix exponential of ``matrix`` using MLX tensors."""

    A = tensor.convert_to_tensor(matrix)
    shape = tensor.shape(A)
    if len(shape) != 2:
        raise ValueError("expm expects a 2-D square matrix")
    if shape[0] != shape[1]:
        raise ValueError("expm expects a square matrix")

    if shape[0] == 0 or shape[1] == 0:
        return tensor.zeros_like(A)

    if shape == (1, 1):
        return ops.exp(A)

    norm_val = ops.norm(A, ord=1)
    s = _compute_scale(float(norm_val))
    scaled = ops.divide(A, 2 ** s)
    U, V = _pade13(scaled)
    X = ops.linearalg.solve(ops.subtract(V, U), ops.add(U, V))

    for _ in range(s):
        X = ops.matmul(X, X)

    return X


def _compute_scale(norm_value: float) -> int:
    if norm_value == 0:
        return 0
    if norm_value <= _THETA_13:
        return 0
    return max(0, math.ceil(math.log2(norm_value / _THETA_13)))


def _pade13(A: tensor.EmberTensor) -> tuple[tensor.EmberTensor, tensor.EmberTensor]:
    n = tensor.shape(A)[0]
    identity = tensor.eye((n, n), dtype=A.dtype)

    A2 = ops.matmul(A, A)
    A4 = ops.matmul(A2, A2)
    A6 = ops.matmul(A4, A2)

    b = _PADE13_COEFFS
    U2 = ops.matmul(A6, ops.add(
        ops.add(ops.multiply(b[13], A6), ops.multiply(b[11], A4)), ops.multiply(b[9], A2)
    ))
    U = ops.matmul(A, ops.add(
        ops.add(ops.add(ops.multiply(b[7], A6), ops.multiply(b[5], A4)), ops.multiply(b[3], A2)),
        ops.multiply(b[1], identity)
    ))

    V2 = ops.matmul(A6, ops.add(
        ops.add(ops.multiply(b[12], A6), ops.multiply(b[10], A4)), ops.multiply(b[8], A2)
    ))
    V = ops.add(
        ops.add(ops.add(ops.multiply(b[6], A6), ops.multiply(b[4], A4)), ops.multiply(b[2], A2)),
        ops.multiply(b[0], identity)
    )

    return U, V
