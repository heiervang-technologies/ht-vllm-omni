# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for Qwen3 TTS model wrapper.

Tests cover:
  - Profile run cap (regression for PR #1082 / #995)
  - Flash-attn detection and fallback
  - Code predictor regional compilation (Phase 1a)

These tests mock heavy dependencies (vllm, transformers, librosa, etc.) so
they can run without GPU, model weights, or the full vllm engine.

The module under test is compiled and executed in a synthetic namespace
to completely bypass the vllm_omni.__init__ import chain.
"""

import atexit
import logging
import sys
import types
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Bootstrap: build a minimal set of stub modules, then compile+exec the
# target .py file in a module whose __package__ resolves relative imports
# to our stubs.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[4]  # repo root
_TARGET = _REPO / "vllm_omni" / "model_executor" / "models" / "qwen3_tts" / "qwen3_tts.py"

# Full set of modules the target file references (directly or via relative
# imports that we intercept).
_STUB_FQNS = [
    # vllm (direct imports in qwen3_tts.py)
    "vllm",
    "vllm.config",
    "vllm.logger",
    "vllm.sequence",
    # transformers (direct import in qwen3_tts.py)
    "transformers",
    # audio I/O libs (direct imports in qwen3_tts.py)
    "librosa",
    "soundfile",
    # vllm_omni package tree (relative imports resolve to these)
    "vllm_omni",
    "vllm_omni.diffusion",
    "vllm_omni.diffusion.compile",
    "vllm_omni.model_executor",
    "vllm_omni.model_executor.models",
    "vllm_omni.model_executor.models.output_templates",
    "vllm_omni.model_executor.models.qwen3_tts",
    "vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts",
    "vllm_omni.model_executor.models.qwen3_tts.modeling_qwen3_tts",
    "vllm_omni.model_executor.models.qwen3_tts.processing_qwen3_tts",
]

_saved_modules: dict[str, types.ModuleType | None] = {}


def _make_stub(fqn: str) -> types.ModuleType:
    parts = fqn.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules:
            mod = types.ModuleType(key)
            mod.__path__ = [str(_REPO / key.replace(".", "/"))]
            mod.__package__ = key
            mod.__spec__ = None
            sys.modules[key] = mod
    return sys.modules[fqn]


_mock_regionally_compile = MagicMock(name="regionally_compile")


def _setup():
    # Save originals so they can be restored on teardown
    for fqn in _STUB_FQNS:
        _saved_modules[fqn] = sys.modules.get(fqn)
        _make_stub(fqn)

    # Wire parent.child attributes
    for fqn in _STUB_FQNS:
        parts = fqn.split(".")
        if len(parts) > 1:
            parent = sys.modules.get(".".join(parts[:-1]))
            child = sys.modules.get(fqn)
            if parent and child:
                setattr(parent, parts[-1], child)

    # ---- Concrete stubs for names the target file actually uses ----

    # vllm.logger
    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)

    # vllm.config
    sys.modules["vllm.config"].VllmConfig = type("VllmConfig", (), {})

    # vllm.sequence
    class IntermediateTensors:
        def __init__(self, d=None):
            self.tensors = d or {}
    sys.modules["vllm.sequence"].IntermediateTensors = IntermediateTensors

    # OmniOutput (from vllm_omni.model_executor.models.output_templates)
    class OmniOutput(NamedTuple):
        text_hidden_states: object
        multimodal_outputs: dict | None = None
        intermediate_tensors: object | None = None
        next_token_id: object | None = None
    sys.modules["vllm_omni.model_executor.models.output_templates"].OmniOutput = OmniOutput

    # vllm_omni.diffusion.compile
    sys.modules["vllm_omni.diffusion.compile"].regionally_compile = _mock_regionally_compile

    # Relative-import siblings
    sys.modules[
        "vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts"
    ].Qwen3TTSConfig = type("Qwen3TTSConfig", (), {})

    sys.modules[
        "vllm_omni.model_executor.models.qwen3_tts.modeling_qwen3_tts"
    ].Qwen3TTSForConditionalGeneration = type("Qwen3TTSForConditionalGeneration", (), {})

    sys.modules[
        "vllm_omni.model_executor.models.qwen3_tts.processing_qwen3_tts"
    ].Qwen3TTSProcessor = type("Qwen3TTSProcessor", (), {})

    # transformers
    sys.modules["transformers"].AutoConfig = MagicMock()
    sys.modules["transformers"].AutoModel = MagicMock()
    sys.modules["transformers"].AutoProcessor = MagicMock()


def _teardown():
    """Restore sys.modules to pre-test state."""
    for fqn in reversed(_STUB_FQNS):
        orig = _saved_modules.get(fqn)
        if orig is None:
            sys.modules.pop(fqn, None)
        else:
            sys.modules[fqn] = orig


_setup()
atexit.register(_teardown)

# Compile and exec the target file in a synthetic module, setting __package__
# so that `from .foo import bar` resolves via sys.modules, not the file system.
_MOD_FQN = "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts"
_mod = types.ModuleType(_MOD_FQN)
_mod.__file__ = str(_TARGET)
_mod.__package__ = "vllm_omni.model_executor.models.qwen3_tts"
_mod.__spec__ = None
sys.modules[_MOD_FQN] = _mod

_source = _TARGET.read_text()
_code = compile(_source, str(_TARGET), "exec")
exec(_code, _mod.__dict__)  # noqa: S102

Qwen3TTSModelForGeneration = _mod.Qwen3TTSModelForGeneration
Qwen3TTSModel = _mod.Qwen3TTSModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_compile_mock():
    """Reset the regionally_compile mock before each test."""
    _mock_regionally_compile.reset_mock()
    _mock_regionally_compile.side_effect = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vllm_config(
    model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    enforce_eager: bool = False,
) -> MagicMock:
    cfg = MagicMock()
    cfg.model_config.model = model_path
    cfg.model_config.enforce_eager = enforce_eager
    return cfg


def _make_wrapper(vllm_config=None):
    """Instantiate the wrapper with a mocked Qwen3TTSModel.from_pretrained."""
    if vllm_config is None:
        vllm_config = _make_vllm_config()

    with patch.object(Qwen3TTSModel, "from_pretrained") as mock_fp:
        mock_fp.return_value = MagicMock()
        wrapper = Qwen3TTSModelForGeneration(vllm_config=vllm_config)

    return wrapper


def _builtins_import():
    import builtins
    return builtins.__import__


def _make_structured_model_mock():
    """Build a mock with explicit attribute hierarchy matching the real model.

    Real hierarchy:
        Qwen3TTSModel (wrapper .model attr)
          +-- Qwen3TTSForConditionalGeneration (.model on wrapper)
                +-- .talker  (Qwen3TTSTalkerForConditionalGeneration)
                      +-- .code_predictor  (Qwen3TTSTalkerCodePredictorModelForConditionalGeneration)
                            +-- .model  (Qwen3TTSTalkerCodePredictorModel -- has .layers)

    Using a structured mock prevents MagicMock from auto-creating wrong paths
    (e.g. .model.model.code_predictor.model without .talker).
    """
    cp_inner_model = MagicMock(name="Qwen3TTSTalkerCodePredictorModel")
    code_predictor = MagicMock(name="CodePredictorForCG")
    code_predictor.model = cp_inner_model

    talker = MagicMock(name="TalkerForCG")
    talker.code_predictor = code_predictor

    hf_model = MagicMock(name="Qwen3TTSForConditionalGeneration")
    hf_model.talker = talker
    # Ensure accessing .code_predictor directly on hf_model raises,
    # so tests would fail if the production code skips .talker
    del hf_model.code_predictor

    wrapper_model = MagicMock(name="Qwen3TTSModel")
    wrapper_model.model = hf_model

    return wrapper_model, cp_inner_model


# ---------------------------------------------------------------------------
# Test A: Profile run cap (regression for PR #1082 / #995)
# ---------------------------------------------------------------------------

class TestProfileRunCap:
    """Empty text caps max_new_tokens and proceeds to generation."""

    def test_empty_text_caps_max_new_tokens(self):
        """Profile run sets max_new_tokens=2 and still calls generation."""
        wrapper = _make_wrapper()
        dummy_wav = np.zeros(24000, dtype=np.float32)
        wrapper.model.generate_voice_clone.return_value = ([dummy_wav], 24000)

        wrapper.forward(
            runtime_additional_information=[{
                "text": [""],
                "task_type": ["Base"],
                "language": ["Auto"],
            }],
        )

        wrapper.model.generate_voice_clone.assert_called_once()
        _, call_kwargs = wrapper.model.generate_voice_clone.call_args
        assert call_kwargs.get("max_new_tokens") == 2

    def test_nonempty_text_proceeds_to_generation(self):
        wrapper = _make_wrapper()
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

    def test_nonempty_text_does_not_cap_max_new_tokens(self):
        """Non-profile runs should not inject max_new_tokens=2."""
        wrapper = _make_wrapper()
        dummy_wav = np.zeros(24000, dtype=np.float32)
        wrapper.model.generate_voice_clone.return_value = ([dummy_wav], 24000)

        wrapper.forward(
            runtime_additional_information=[{
                "text": ["Hello world"],
                "task_type": ["Base"],
                "language": ["Auto"],
            }],
        )

        _, call_kwargs = wrapper.model.generate_voice_clone.call_args
        assert call_kwargs.get("max_new_tokens") != 2


# ---------------------------------------------------------------------------
# Test B: Flash-attn detection
# ---------------------------------------------------------------------------

class TestFlashAttnDetection:
    """Verify attn_implementation kwarg passed to Qwen3TTSModel.from_pretrained."""

    def test_no_attn_kwarg_without_flash_attn(self):
        """When flash_attn is not importable, from_pretrained gets no attn_implementation."""
        saved = sys.modules.pop("flash_attn", None)
        try:
            real_import = _builtins_import()

            def _fake_import(name, *args, **kwargs):
                if name == "flash_attn":
                    raise ImportError("mocked: no flash_attn")
                return real_import(name, *args, **kwargs)

            vllm_config = _make_vllm_config()
            with (
                patch("builtins.__import__", side_effect=_fake_import),
                patch.object(Qwen3TTSModel, "from_pretrained") as mock_fp,
            ):
                mock_fp.return_value = MagicMock()
                Qwen3TTSModelForGeneration(vllm_config=vllm_config)

                mock_fp.assert_called_once()
                _, call_kwargs = mock_fp.call_args
                assert "attn_implementation" not in call_kwargs, \
                    f"Expected no attn_implementation kwarg, got: {call_kwargs}"
        finally:
            if saved is not None:
                sys.modules["flash_attn"] = saved

    def test_flash_attn_preferred_when_available(self):
        """When flash_attn is importable, from_pretrained receives flash_attention_2."""
        vllm_config = _make_vllm_config()

        fake_flash = types.ModuleType("flash_attn")
        sys.modules["flash_attn"] = fake_flash
        try:
            with patch.object(Qwen3TTSModel, "from_pretrained") as mock_fp:
                mock_fp.return_value = MagicMock()
                Qwen3TTSModelForGeneration(vllm_config=vllm_config)

                mock_fp.assert_called_once()
                _, call_kwargs = mock_fp.call_args
                assert call_kwargs.get("attn_implementation") == "flash_attention_2", \
                    f"Expected attn_implementation='flash_attention_2', got: {call_kwargs}"
        finally:
            sys.modules.pop("flash_attn", None)


# ---------------------------------------------------------------------------
# Test C: Code predictor regional compilation (Phase 1a)
# ---------------------------------------------------------------------------

class TestCodePredictorCompilation:
    """Verify regionally_compile is called on the code predictor model."""

    def test_regionally_compile_called_on_init(self):
        """regionally_compile is called with the code predictor's inner model and dynamic=True."""
        vllm_config = _make_vllm_config()
        wrapper_model, cp_inner = _make_structured_model_mock()
        with patch.object(Qwen3TTSModel, "from_pretrained") as mock_fp:
            mock_fp.return_value = wrapper_model
            Qwen3TTSModelForGeneration(vllm_config=vllm_config)

        _mock_regionally_compile.assert_called_once()
        call_args, call_kwargs = _mock_regionally_compile.call_args
        assert call_args[0] is cp_inner
        assert call_kwargs.get("dynamic") is True
        assert "mode" not in call_kwargs

    def test_repeated_blocks_set_before_compile(self):
        """_repeated_blocks attribute is set on the code predictor model."""
        vllm_config = _make_vllm_config()
        wrapper_model, cp_inner = _make_structured_model_mock()
        with patch.object(Qwen3TTSModel, "from_pretrained") as mock_fp:
            mock_fp.return_value = wrapper_model
            Qwen3TTSModelForGeneration(vllm_config=vllm_config)

        assert cp_inner._repeated_blocks == ["Qwen3TTSDecoderLayer"]

    def test_compile_failure_does_not_crash(self):
        """If regionally_compile raises RuntimeError, __init__ still succeeds."""
        _mock_regionally_compile.side_effect = RuntimeError("compile failed")

        vllm_config = _make_vllm_config()
        wrapper_model, _ = _make_structured_model_mock()
        with patch.object(Qwen3TTSModel, "from_pretrained") as mock_fp:
            mock_fp.return_value = wrapper_model
            wrapper = Qwen3TTSModelForGeneration(vllm_config=vllm_config)
            assert wrapper is not None

    def test_enforce_eager_skips_compilation(self):
        """When enforce_eager=True, regionally_compile is not called."""
        vllm_config = _make_vllm_config(enforce_eager=True)
        wrapper_model, _ = _make_structured_model_mock()
        with patch.object(Qwen3TTSModel, "from_pretrained") as mock_fp:
            mock_fp.return_value = wrapper_model
            Qwen3TTSModelForGeneration(vllm_config=vllm_config)

        _mock_regionally_compile.assert_not_called()
