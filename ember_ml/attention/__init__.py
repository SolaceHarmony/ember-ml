"""Compatibility wrapper exposing attention modules."""

from ember_ml.nn.attention import *
import ember_ml.nn.attention as _nn_attention
from ember_ml.models.attention.multiscale_ltc import (
    TemporalStrideProcessor,
    build_multiscale_ltc_model,
    visualize_feature_extraction,
    visualize_multiscale_dynamics,
)

__all__ = list(getattr(_nn_attention, "__all__", [])) + [
    "TemporalStrideProcessor",
    "build_multiscale_ltc_model",
    "visualize_feature_extraction",
    "visualize_multiscale_dynamics",
]
