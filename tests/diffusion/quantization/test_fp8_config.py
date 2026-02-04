# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for diffusion FP8 quantization config and transformer wiring."""

from __future__ import annotations

import pytest


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Config-layer tests (no GPU needed)
# ---------------------------------------------------------------------------


class TestDiffusionQuantizationRegistry:
    """Test the quantization registry and factory functions."""

    def test_get_diffusion_quant_config_fp8(self):
        from vllm_omni.diffusion.quantization import get_diffusion_quant_config

        cfg = get_diffusion_quant_config("fp8")
        assert cfg is not None
        assert cfg.get_name() == "fp8"

    def test_get_diffusion_quant_config_none(self):
        from vllm_omni.diffusion.quantization import get_diffusion_quant_config

        assert get_diffusion_quant_config(None) is None

    def test_get_diffusion_quant_config_unknown_raises(self):
        from vllm_omni.diffusion.quantization import get_diffusion_quant_config

        with pytest.raises(ValueError, match="Unknown diffusion quantization"):
            get_diffusion_quant_config("nonexistent_method")

    def test_vllm_bridge(self):
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        from vllm_omni.diffusion.quantization import (
            get_diffusion_quant_config,
            get_vllm_quant_config_for_layers,
        )

        diff_cfg = get_diffusion_quant_config("fp8")
        vllm_cfg = get_vllm_quant_config_for_layers(diff_cfg)
        assert isinstance(vllm_cfg, Fp8Config)
        assert vllm_cfg.is_checkpoint_fp8_serialized is False
        assert vllm_cfg.activation_scheme == "dynamic"

    def test_vllm_bridge_none(self):
        from vllm_omni.diffusion.quantization import get_vllm_quant_config_for_layers

        assert get_vllm_quant_config_for_layers(None) is None


class TestDiffusionFp8Config:
    """Test DiffusionFp8Config construction."""

    def test_default_activation_scheme(self):
        from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

        cfg = DiffusionFp8Config()
        vllm = cfg.get_vllm_quant_config()
        assert vllm.activation_scheme == "dynamic"

    def test_custom_activation_scheme(self):
        from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

        cfg = DiffusionFp8Config(activation_scheme="static")
        vllm = cfg.get_vllm_quant_config()
        assert vllm.activation_scheme == "static"

    def test_repr(self):
        from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

        cfg = DiffusionFp8Config()
        r = repr(cfg)
        assert "DiffusionFp8Config" in r


class TestOmniDiffusionConfigQuantization:
    """Test OmniDiffusionConfig quantization field resolution."""

    def test_string_quantization(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

        c = OmniDiffusionConfig(quantization="fp8")
        assert isinstance(c.quantization_config, DiffusionFp8Config)
        assert c.quantization == "fp8"

    def test_no_quantization(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        c = OmniDiffusionConfig()
        assert c.quantization is None
        assert c.quantization_config is None

    def test_dict_quantization_config(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

        c = OmniDiffusionConfig(
            quantization_config={"method": "fp8", "activation_scheme": "dynamic"}
        )
        assert isinstance(c.quantization_config, DiffusionFp8Config)

    def test_dict_with_quantization_string(self):
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

        c = OmniDiffusionConfig(
            quantization="fp8",
            quantization_config={"activation_scheme": "dynamic"},
        )
        assert isinstance(c.quantization_config, DiffusionFp8Config)
        assert c.quantization == "fp8"


# ---------------------------------------------------------------------------
# Transformer wiring test (needs GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_cuda(),
    reason="No CUDA device available",
)
class TestQwenImageFp8Wiring:
    """Verify that FP8 quant_method is attached to linear layers."""

    def test_transformer_has_quant_methods(self):
        import torch

        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.quantization import get_vllm_quant_config_for_layers
        from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
            QwenImageTransformer2DModel,
        )

        od_config = OmniDiffusionConfig(quantization="fp8")
        quant_config = get_vllm_quant_config_for_layers(od_config.quantization_config)

        with torch.device("meta"):
            model = QwenImageTransformer2DModel(
                od_config=od_config,
                quant_config=quant_config,
                num_layers=2,  # minimal for fast test
                num_attention_heads=4,
                attention_head_dim=32,
                in_channels=16,
                out_channels=16,
                joint_attention_dim=128,
            )

        # Check that at least some modules got a quant_method
        modules_with_quant = [
            name
            for name, mod in model.named_modules()
            if hasattr(mod, "quant_method") and mod.quant_method is not None
        ]
        assert len(modules_with_quant) > 0, (
            "No modules have quant_method. FP8 wiring is broken."
        )

    def test_transformer_without_quant_has_no_quant_methods(self):
        import torch

        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
            QwenImageTransformer2DModel,
        )

        od_config = OmniDiffusionConfig()

        with torch.device("meta"):
            model = QwenImageTransformer2DModel(
                od_config=od_config,
                num_layers=2,
                num_attention_heads=4,
                attention_head_dim=32,
                in_channels=16,
                out_channels=16,
                joint_attention_dim=128,
            )

        modules_with_quant = [
            name
            for name, mod in model.named_modules()
            if hasattr(mod, "quant_method") and mod.quant_method is not None
        ]
        assert len(modules_with_quant) == 0, (
            f"Unexpected quant_method on modules without quantization: {modules_with_quant}"
        )
