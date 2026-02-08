# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for diffusion model quantization configurations."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

import torch
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)


class DiffusionQuantizationConfig(ABC):
    """Abstract base class wrapping a vLLM ``QuantizationConfig``.

    Subclasses set ``quant_config_cls`` to the concrete vLLM config class
    (e.g. ``Fp8Config``) and build the underlying instance in ``__init__``.
    """

    quant_config_cls: ClassVar[type[QuantizationConfig]]

    # Underlying vLLM config created by subclass __init__
    _vllm_config: QuantizationConfig

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> str:
        return cls.quant_config_cls.get_name()

    def get_vllm_quant_config(self) -> QuantizationConfig:
        """Return the underlying vLLM ``QuantizationConfig``."""
        return self._vllm_config

    @staticmethod
    def get_supported_act_dtypes() -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        if hasattr(cls.quant_config_cls, "get_min_capability"):
            return cls.quant_config_cls.get_min_capability()
        return 80  # Ampere+

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vllm_config={self._vllm_config!r})"
