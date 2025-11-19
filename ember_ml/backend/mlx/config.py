"""
MLX backend configuration for ember_ml.

This module provides configuration settings for the MLX backend.
"""

import mlx.core as mx

from ember_ml.backend.mlx.tensor.dtype import MLXDType


def _normalize_device_name(device: object) -> str:
    """Return a canonical device string for MLX devices."""

    if isinstance(device, str):
        name = device.lower()
    else:
        dev_type = None
        if hasattr(device, "type"):
            dev_type = getattr(device, "type")
        if dev_type is None and hasattr(device, "name"):
            name = str(getattr(device, "name")).lower()
        elif dev_type is not None:
            name_attr = getattr(dev_type, "name", None)
            if isinstance(name_attr, str):
                name = name_attr.lower()
            else:
                name = str(dev_type).lower()
        else:
            name = str(device).lower()

    if name.startswith("devicetype."):
        name = name.split(".", 1)[1]

    if "gpu" in name or name in {"metal", "device(gpu)"}:
        return "gpu"
    if "cpu" in name:
        return "cpu"

    return "cpu"


def _detect_default_device() -> str:
    """Detect the best default device available for MLX."""

    try:
        default = mx.default_device()
    except Exception:
        return "cpu"

    try:
        return _normalize_device_name(default)
    except Exception:
        return "cpu"


# Default device for MLX operations. MLX exposes CPU tensors even when Metal is
# unavailable, so fall back to ``"cpu"`` when GPU initialisation fails.
DEFAULT_DEVICE = _detect_default_device()

# Default data type for MLX operations
DEFAULT_DTYPE = MLXDType().float32


__all__ = ["DEFAULT_DEVICE", "DEFAULT_DTYPE"]
