# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion model quantization support.

Provides a thin wrapper around vLLM's quantization infrastructure so that
diffusion transformer layers can opt-in to weight/activation quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.diffusion.quantization.base import DiffusionQuantizationConfig
from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)

# -----------------------------------------------------------------
# Registry
# -----------------------------------------------------------------

_QUANT_CONFIG_REGISTRY: dict[str, type[DiffusionQuantizationConfig]] = {
    "fp8": DiffusionFp8Config,
}

SUPPORTED_QUANTIZATION_METHODS: list[str] = list(_QUANT_CONFIG_REGISTRY.keys())


# -----------------------------------------------------------------
# Factory
# -----------------------------------------------------------------


def get_diffusion_quant_config(
    quantization: str | None,
    **kwargs,
) -> DiffusionQuantizationConfig | None:
    """Create a ``DiffusionQuantizationConfig`` by method name.

    Args:
        quantization: Quantization method name (e.g. ``"fp8"``).
            ``None`` means no quantization.
        **kwargs: Extra keyword arguments forwarded to the config constructor
            (e.g. ``activation_scheme``, ``ignored_layers``).

    Returns:
        A config instance, or ``None`` when *quantization* is ``None``.

    Raises:
        ValueError: If the method name is not in the registry.
    """
    if quantization is None:
        return None

    quantization = quantization.lower()
    if quantization not in _QUANT_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown diffusion quantization method {quantization!r}. "
            f"Supported methods: {SUPPORTED_QUANTIZATION_METHODS}"
        )

    config_cls = _QUANT_CONFIG_REGISTRY[quantization]
    return config_cls(**kwargs)


def get_vllm_quant_config_for_layers(
    diffusion_quant_config: DiffusionQuantizationConfig | None,
) -> "QuantizationConfig | None":
    """Extract the underlying vLLM ``QuantizationConfig`` for passing to layers.

    This is the bridge between the diffusion quantization world and
    vLLM's parallel-linear layers which accept ``quant_config``.

    Args:
        diffusion_quant_config: A ``DiffusionQuantizationConfig`` instance,
            or ``None`` for no quantization.

    Returns:
        A vLLM ``QuantizationConfig`` (e.g. ``Fp8Config``), or ``None``.
    """
    if diffusion_quant_config is None:
        return None
    return diffusion_quant_config.get_vllm_quant_config()
