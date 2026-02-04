#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test for FP8 quantization of diffusion models.

Bypasses the heavy vllm_omni.__init__ import chain by patching sys.modules
before importing the diffusion subpackage directly.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tests/diffusion/quantization/test_fp8_smoke.py
"""

from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Stub the top-level vllm_omni package to prevent __init__.py from loading
# the full entrypoints/distributed stack (which requires omegaconf etc.)
# ---------------------------------------------------------------------------

_stub = types.ModuleType("vllm_omni")
_stub.__path__ = ["vllm_omni"]
_stub.__package__ = "vllm_omni"
sys.modules.setdefault("vllm_omni", _stub)


def _banner(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def _ensure_vllm_config():
    """Set up minimal vLLM config and distributed state for single-GPU testing."""
    import os

    import torch
    import vllm.config.vllm as vllm_config_mod
    from vllm.config.vllm import VllmConfig

    if vllm_config_mod._current_vllm_config is not None:
        return  # already set

    vllm_config_mod._current_vllm_config = VllmConfig()
    vllm_config_mod.get_cached_compilation_config.cache_clear()

    # Initialize torch.distributed for single GPU (needed by parallel linear layers)
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)

    # Initialize vLLM distributed state for TP=1
    import vllm.distributed.parallel_state as ps

    if ps._WORLD is None:
        ps._WORLD = ps.init_world_group(ranks=[0], local_rank=0, backend="gloo")

    if ps._TP is None:
        ps.initialize_model_parallel(tensor_model_parallel_size=1)


def test_quantization_config_factory():
    """Test that the config registry and factory work correctly."""
    _banner("Test 1: Quantization config factory")

    from vllm_omni.diffusion.quantization import (
        get_diffusion_quant_config,
        get_vllm_quant_config_for_layers,
    )
    from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

    # Factory creates correct type
    cfg = get_diffusion_quant_config("fp8")
    assert isinstance(cfg, DiffusionFp8Config), f"Expected DiffusionFp8Config, got {type(cfg)}"
    print(f"  [PASS] get_diffusion_quant_config('fp8') -> {cfg!r}")

    # None -> None
    assert get_diffusion_quant_config(None) is None
    print("  [PASS] get_diffusion_quant_config(None) -> None")

    # Unknown -> ValueError
    try:
        get_diffusion_quant_config("fake")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  [PASS] Unknown method raises ValueError")

    # Bridge extracts vLLM Fp8Config
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    vllm_cfg = get_vllm_quant_config_for_layers(cfg)
    assert isinstance(vllm_cfg, Fp8Config)
    assert vllm_cfg.is_checkpoint_fp8_serialized is False
    assert vllm_cfg.activation_scheme == "dynamic"
    print("  [PASS] vLLM bridge -> Fp8Config(serialized=False, scheme=dynamic)")

    # Bridge with None
    assert get_vllm_quant_config_for_layers(None) is None
    print("  [PASS] vLLM bridge(None) -> None")


def test_omni_diffusion_config_quantization():
    """Test OmniDiffusionConfig quantization field resolution."""
    _banner("Test 2: OmniDiffusionConfig quantization fields")

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.quantization.fp8 import DiffusionFp8Config

    # String -> config
    c = OmniDiffusionConfig(quantization="fp8")
    assert isinstance(c.quantization_config, DiffusionFp8Config)
    assert c.quantization == "fp8"
    print("  [PASS] quantization='fp8' -> DiffusionFp8Config")

    # No quantization -> None
    c2 = OmniDiffusionConfig()
    assert c2.quantization is None
    assert c2.quantization_config is None
    print("  [PASS] No quantization -> None")

    # Dict config
    c3 = OmniDiffusionConfig(
        quantization_config={"method": "fp8", "activation_scheme": "dynamic"}
    )
    assert isinstance(c3.quantization_config, DiffusionFp8Config)
    print("  [PASS] Dict quantization_config -> DiffusionFp8Config")

    # String + dict combo
    c4 = OmniDiffusionConfig(
        quantization="fp8",
        quantization_config={"activation_scheme": "dynamic"},
    )
    assert isinstance(c4.quantization_config, DiffusionFp8Config)
    assert c4.quantization == "fp8"
    print("  [PASS] quantization='fp8' + dict -> DiffusionFp8Config")


def test_qwen_image_transformer_fp8_wiring():
    """Instantiate QwenImageTransformer2DModel with FP8 and verify quant_method."""
    _banner("Test 3: QwenImageTransformer2DModel FP8 wiring (meta device)")

    import torch

    _ensure_vllm_config()

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.quantization import get_vllm_quant_config_for_layers
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )

    od_config = OmniDiffusionConfig(quantization="fp8")
    quant_config = get_vllm_quant_config_for_layers(od_config.quantization_config)

    # Use meta device so no GPU memory is allocated
    with torch.device("meta"):
        model = QwenImageTransformer2DModel(
            od_config=od_config,
            quant_config=quant_config,
            num_layers=2,
            num_attention_heads=4,
            attention_head_dim=32,
            in_channels=16,
            out_channels=16,
            joint_attention_dim=128,
        )

    # Collect modules with quant_method
    quantized_modules = []
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is not None:
            quantized_modules.append(name)

    print(f"  Found {len(quantized_modules)} modules with quant_method:")
    for name in quantized_modules:
        print(f"    - {name}")

    assert len(quantized_modules) > 0, "No modules have quant_method -- FP8 wiring is broken"
    print(f"  [PASS] {len(quantized_modules)} modules have quant_method")


def test_qwen_image_transformer_no_quant_wiring():
    """Verify that without quantization, no FP8 quant_method is attached."""
    _banner("Test 4: QwenImageTransformer2DModel without quantization")

    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

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

    # When no quant_config is passed, vLLM assigns UnquantizedLinearMethod.
    # Check that NO module has a non-Unquantized quant_method.
    fp8_quantized = [
        name
        for name, mod in model.named_modules()
        if hasattr(mod, "quant_method")
        and mod.quant_method is not None
        and not isinstance(mod.quant_method, UnquantizedLinearMethod)
    ]
    assert len(fp8_quantized) == 0, (
        f"Unexpected FP8 quant_method on {fp8_quantized}"
    )
    print("  [PASS] No modules have FP8 quant_method (all UnquantizedLinearMethod)")


def test_fp8_forward_on_gpu():
    """Actually run a forward pass with FP8 quantized weights on GPU."""
    _banner("Test 5: FP8 forward pass on GPU")

    import torch

    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA device available")
        return

    # FP8 compute requires SM 89+ (Ada Lovelace / Hopper)
    capability = torch.cuda.get_device_capability()
    if capability < (8, 9):
        print(
            f"  [SKIP] GPU compute capability {capability[0]}.{capability[1]} < 8.9 "
            f"-- native FP8 requires Ada Lovelace (RTX 4090) or Hopper (H100)+"
        )
        return

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.quantization import get_vllm_quant_config_for_layers
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
        QwenImageTransformer2DModel,
    )
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    device = torch.device("cuda")
    od_config = OmniDiffusionConfig(quantization="fp8")
    quant_config = get_vllm_quant_config_for_layers(od_config.quantization_config)

    # Build a tiny model on GPU with random weights
    with torch.device(device):
        model = QwenImageTransformer2DModel(
            od_config=od_config,
            quant_config=quant_config,
            num_layers=2,
            num_attention_heads=4,
            attention_head_dim=32,
            in_channels=16,
            out_channels=16,
            joint_attention_dim=128,
        )

    # Initialize random weights (since we're not loading from checkpoint)
    for param in model.parameters():
        if param.requires_grad:
            torch.nn.init.normal_(param, std=0.02)

    # Run post-loading weight processing (this is where FP8 conversion happens)
    DiffusersPipelineLoader._process_weights_after_loading(model, device)

    # Count modules that were quantized
    quantized_modules = [
        name
        for name, mod in model.named_modules()
        if getattr(mod, "quant_method", None) is not None
    ]
    print(f"  {len(quantized_modules)} modules quantized to FP8")

    # Check that at least some weights are FP8
    fp8_params = []
    for name, param in model.named_parameters():
        if param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            fp8_params.append((name, param.dtype))

    print(f"  {len(fp8_params)} parameters converted to FP8:")
    for name, dtype in fp8_params[:10]:
        print(f"    - {name}: {dtype}")
    if len(fp8_params) > 10:
        print(f"    ... and {len(fp8_params) - 10} more")

    # Forward pass smoke test
    # We need to set up the forward context for QwenImage
    from vllm_omni.diffusion.forward_context import set_forward_context

    batch_size = 1
    img_seq_len = 16  # 4x4 patch grid -> 1 frame, 4 height, 4 width
    txt_seq_len = 8
    hidden_dim = 16  # in_channels

    hidden_states = torch.randn(batch_size, img_seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, 128, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
    img_shapes = [[(1, 4, 4)]]
    txt_seq_lens = [txt_seq_len]

    with set_forward_context(omni_diffusion_config=od_config):
        model.eval()
        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )

    sample = output.sample
    print(f"  Output shape: {sample.shape}")
    assert sample.shape == (batch_size, img_seq_len, 16 * 4), (
        f"Unexpected output shape: {sample.shape}"
    )
    assert not torch.isnan(sample).any(), "Output contains NaN!"
    assert not torch.isinf(sample).any(), "Output contains Inf!"
    print("  [PASS] Forward pass completed without errors")

    # Memory comparison
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_quantization_config_factory()
    test_omni_diffusion_config_quantization()
    test_qwen_image_transformer_fp8_wiring()
    test_qwen_image_transformer_no_quant_wiring()
    test_fp8_forward_on_gpu()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
