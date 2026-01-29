# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for Qwen3 TTS model wrapper.

Tests cover:
  - Profile run short-circuit (regression for PR #1082)
  - SDPA attention fallback when flash-attn is unavailable

These tests mock heavy dependencies (vllm, transformers, librosa, etc.) so
they can run without GPU, model weights, or the full vllm engine.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Module-level mocking: stub out vllm and heavy transitive deps so that
# vllm_omni.model_executor.models.qwen3_tts.qwen3_tts can be imported
# without the full vllm engine or HuggingFace weights.
# ---------------------------------------------------------------------------

def _ensure_mock_module(fqn: str) -> MagicMock:
    """Insert *fqn* (and every parent) into sys.modules as a MagicMock."""
    parts = fqn.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules or not isinstance(sys.modules[key], (MagicMock, types.ModuleType)):
            sys.modules[key] = MagicMock()
    return sys.modules[fqn]


# Modules that the import chain needs but are either heavy or absent in CI.
_MOCK_MODULES = [
    # vllm core
    "vllm",
    "vllm.config",
    "vllm.logger",
    "vllm.sequence",
    "vllm.distributed",
    "vllm.distributed.parallel_state",
    "vllm.utils",
    "vllm.utils.network_utils",
    # vllm_omni.config chain
    "vllm.config.lora",
    # transformers (needed by from_pretrained helpers)
    "transformers",
    # librosa / soundfile — used by audio helpers
    "librosa",
    "soundfile",
]

_original_modules: dict[str, types.ModuleType | None] = {}


def _install_mocks():
    for mod in _MOCK_MODULES:
        _original_modules[mod] = sys.modules.get(mod)
        _ensure_mock_module(mod)

    # Make vllm.logger.init_logger return a stdlib-compatible logger mock
    import logging
    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)

    # Provide a real IntermediateTensors stub so OmniOutput NamedTuple works
    mock_seq = sys.modules["vllm.sequence"]
    mock_seq.IntermediateTensors = type("IntermediateTensors", (), {"__init__": lambda self, d: None})


def _uninstall_mocks():
    for mod, orig in _original_modules.items():
        if orig is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = orig
    _original_modules.clear()

    # Remove cached vllm_omni sub-modules so next import is clean
    to_remove = [k for k in sys.modules if k.startswith("vllm_omni")]
    for k in to_remove:
        del sys.modules[k]


# Install mocks at module load time (before any test collects)
_install_mocks()


# Now safe to import the module under test
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import (  # noqa: E402
    Qwen3TTSModelForGeneration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vllm_config(model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base") -> MagicMock:
    """Create a minimal VllmConfig mock."""
    cfg = MagicMock()
    cfg.model_config.model = model_path
    return cfg


def _make_wrapper(vllm_config=None):
    """Instantiate Qwen3TTSModelForGeneration with a mocked from_pretrained."""
    if vllm_config is None:
        vllm_config = _make_vllm_config()

    with patch(
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts.Qwen3TTSModel.from_pretrained"
    ) as mock_fp:
        mock_fp.return_value = MagicMock()
        wrapper = Qwen3TTSModelForGeneration(vllm_config=vllm_config)

    return wrapper


# ---------------------------------------------------------------------------
# Test A: Profile run short-circuit (regression for PR #1082)
# ---------------------------------------------------------------------------

class TestProfileRunShortCircuit:
    """Empty text triggers a dummy audio return instead of hanging."""

    def test_empty_text_returns_dummy_audio(self):
        wrapper = _make_wrapper()
        result = wrapper.forward(
            runtime_additional_information=[{"text": [""]}],
        )

        assert result.multimodal_outputs is not None
        audio = result.multimodal_outputs["model_outputs"]
        assert audio.shape == (24000,)
        assert result.multimodal_outputs["sr"].item() == 24000

    def test_empty_text_skips_generation(self):
        wrapper = _make_wrapper()
        wrapper.forward(
            runtime_additional_information=[{"text": [""]}],
        )

        model = wrapper.model
        model.generate_voice_clone.assert_not_called()
        model.generate_custom_voice.assert_not_called()
        model.generate_voice_design.assert_not_called()

    def test_nonempty_text_proceeds_to_generation(self):
        wrapper = _make_wrapper()
        # generate_voice_clone returns (list[ndarray], int)
        import numpy as np

        dummy_wav = np.zeros(24000, dtype=np.float32)
        wrapper.model.generate_voice_clone.return_value = ([dummy_wav], 24000)

        wrapper.forward(
            runtime_additional_information=[{
                "text": ["Hello"],
                "task_type": ["Base"],
                "language": ["Auto"],
            }],
        )

        wrapper.model.generate_voice_clone.assert_called_once()


# ---------------------------------------------------------------------------
# Test B: SDPA attention fallback
# ---------------------------------------------------------------------------

class TestSDPAFallback:
    """Verify attn_implementation kwarg passed to from_pretrained."""

    def test_sdpa_fallback_without_flash_attn(self):
        """When flash_attn is not importable, from_pretrained receives sdpa."""
        # Remove flash_attn from sys.modules if present
        saved = sys.modules.pop("flash_attn", None)
        try:
            real_import = builtins_import()

            def _fake_import(name, *args, **kwargs):
                if name == "flash_attn":
                    raise ImportError("mocked: no flash_attn")
                return real_import(name, *args, **kwargs)

            vllm_config = _make_vllm_config()
            with (
                patch(
                    "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts.Qwen3TTSModel.from_pretrained"
                ) as mock_fp,
                patch("builtins.__import__", side_effect=_fake_import),
            ):
                mock_fp.return_value = MagicMock()
                # Reload to re-execute the try/except import block
                import vllm_omni.model_executor.models.qwen3_tts.qwen3_tts as mod
                importlib.reload(mod)
                mod.Qwen3TTSModelForGeneration(vllm_config=vllm_config)

                mock_fp.assert_called_once()
                _, call_kwargs = mock_fp.call_args
                assert call_kwargs.get("attn_implementation") == "sdpa", \
                    f"Expected attn_implementation='sdpa', got: {call_kwargs}"
        finally:
            if saved is not None:
                sys.modules["flash_attn"] = saved

    def test_flash_attn_preferred_when_available(self):
        """When flash_attn is importable, from_pretrained receives flash_attention_2."""
        vllm_config = _make_vllm_config()

        fake_flash = MagicMock()
        with (
            patch(
                "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts.Qwen3TTSModel.from_pretrained"
            ) as mock_fp,
            patch.dict("sys.modules", {"flash_attn": fake_flash}),
        ):
            mock_fp.return_value = MagicMock()
            import vllm_omni.model_executor.models.qwen3_tts.qwen3_tts as mod
            importlib.reload(mod)
            mod.Qwen3TTSModelForGeneration(vllm_config=vllm_config)

            mock_fp.assert_called_once()
            _, call_kwargs = mock_fp.call_args
            assert call_kwargs.get("attn_implementation") == "flash_attention_2", \
                f"Expected attn_implementation='flash_attention_2', got: {call_kwargs}"


def builtins_import():
    """Get the real __import__ function."""
    import builtins
    return builtins.__import__
