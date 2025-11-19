"""PyTorch backend matrix exponential via scaling-and-squaring + Pade-13."""

from __future__ import annotations

import math
from typing import Tuple

import torch

from ember_ml.backend.torch.tensor.tensor import TorchTensor

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


def expm(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the matrix exponential using Pade-13 scaling and squaring."""

    tensor_helper = TorchTensor()
    array = tensor_helper.convert(matrix)

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("expm expects a square 2-D tensor")

    result = _compute_expm(array)
    return result


def _compute_expm(array: torch.Tensor) -> torch.Tensor:
    if array.numel() == 0:
        return torch.zeros_like(array)

    if array.shape == torch.Size([1, 1]):
        return torch.exp(array)

    col_sums = torch.sum(torch.abs(array), dim=0)
    norm_value = float(torch.max(col_sums).item())
    s = _compute_scale(norm_value)
    scaled = array / (2 ** s)
    U, V = _pade13(scaled)
    X = torch.linalg.solve(V - U, U + V)
    for _ in range(s):
        X = X @ X
    return X


def _compute_scale(norm_value: float) -> int:
    if norm_value == 0:
        return 0
    if norm_value <= _THETA_13:
        return 0
    return max(0, math.ceil(math.log2(norm_value / _THETA_13)))


def _pade13(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = A.shape[0]
    identity = torch.eye(n, dtype=A.dtype, device=A.device)

    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2

    b = _PADE13_COEFFS
    U2 = A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
    U = A @ (U2 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * identity)

    V2 = A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
    V = V2 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * identity

    return U, V
