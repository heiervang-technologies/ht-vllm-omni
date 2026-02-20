# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization configuration for diffusion models."""

from __future__ import annotations

from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_omni.diffusion.quantization.base import DiffusionQuantizationConfig


class DiffusionFp8Config(DiffusionQuantizationConfig):
    """FP8 quantization for diffusion transformers.

    Creates an ``Fp8Config`` configured for *online* quantization:
    weights are stored in BF16/FP16 and converted to FP8 at runtime
    (``is_checkpoint_fp8_serialized=False``).
    """

    quant_config_cls = Fp8Config

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        weight_block_size: list[int] | None = None,
        ignored_layers: list[str] | None = None,
    ) -> None:
        self._vllm_config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme=activation_scheme,
            weight_block_size=weight_block_size,
            ignored_layers=ignored_layers,
        )
