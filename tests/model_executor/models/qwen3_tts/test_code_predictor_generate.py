# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for code predictor generate_codes() and _sample_token().

Tests cover:
  - _sample_token: greedy, temperature, top-k, top-p, output shape
  - generate_codes: output shape, determinism, None kwargs, valid range
  - Regression: generate_codes() matches HF generate() under greedy decoding

These tests stub only the vllm-specific dependencies so that the real
transformers-based modeling code can be imported and exercised with small
random-weight models.
"""

import logging
import sys
import types
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Bootstrap: stub vllm-specific modules that the modeling code imports,
# then import the real transformers-backed classes.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[4]
_saved: dict[str, types.ModuleType | None] = {}

# Modules that need stubs (vllm / vllm_omni internals not available in test).
# The full vllm_omni package chain must be stubbed to prevent the real
# __init__.py files from executing (they pull in config, entrypoints, etc.).
_STUB_FQNS = [
    "vllm",
    "vllm.config",
    "vllm.logger",
    "vllm.sequence",
    # vllm_omni package chain — prevent real __init__.py cascade
    "vllm_omni",
    "vllm_omni.patch",
    "vllm_omni.model_executor",
    "vllm_omni.model_executor.model_loader",
    "vllm_omni.model_executor.model_loader.weight_utils",
    "vllm_omni.model_executor.models",
    "vllm_omni.model_executor.models.qwen3_omni",
    "vllm_omni.model_executor.models.registry",
    "vllm_omni.model_executor.models.qwen3_tts",
]


def _setup_stubs():
    """Pre-seed sys.modules with minimal stubs for vllm dependencies."""
    for fqn in _STUB_FQNS:
        _saved[fqn] = sys.modules.get(fqn)
        if fqn not in sys.modules:
            mod = types.ModuleType(fqn)
            mod.__path__ = [str(_REPO / fqn.replace(".", "/"))]
            mod.__package__ = fqn
            mod.__spec__ = None
            sys.modules[fqn] = mod

    # Concrete stubs for names the target code actually uses
    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)
    sys.modules["vllm.config"].VllmConfig = type("VllmConfig", (), {})

    weight_mod = sys.modules["vllm_omni.model_executor.model_loader.weight_utils"]
    weight_mod.download_weights_from_hf_specific = lambda *a, **kw: None

    # models/__init__.py expects these names
    sys.modules["vllm_omni.model_executor.models.qwen3_omni"].Qwen3OmniMoeForConditionalGeneration = type(
        "Stub", (), {}
    )
    sys.modules["vllm_omni.model_executor.models.registry"].OmniModelRegistry = type("Stub", (), {})

    # Patch ROPE_INIT_FUNCTIONS — transformers 5.x removed 'default';
    # the modeling code still references it.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope(config=None, device=None, seq_len=None, layer_type=None):
            base = getattr(config, "rope_theta", 10000.0)
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
            )
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _default_rope

    # Wire parent→child attributes
    for fqn in _STUB_FQNS:
        parts = fqn.split(".")
        if len(parts) > 1:
            parent = sys.modules.get(".".join(parts[:-1]))
            child = sys.modules.get(fqn)
            if parent and child:
                setattr(parent, parts[-1], child)


_setup_stubs()

# Now import the real modeling code — all transformers imports resolve normally.
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (  # noqa: E402
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.modeling_qwen3_tts import (  # noqa: E402
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    _sample_token,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Small model dimensions for fast tests
_VOCAB = 64
_HIDDEN = 32
_INTERMEDIATE = 64
_HEADS = 4
_KV_HEADS = 2
_LAYERS = 1
_CODE_GROUPS = 4  # generates 3 codes per invocation


@pytest.fixture()
def code_predictor():
    """Build a tiny code predictor with random weights."""
    cp_config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        num_code_groups=_CODE_GROUPS,
        max_position_embeddings=128,
        use_cache=True,
        pad_token_id=0,
    )
    talker_config = Qwen3TTSTalkerConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        num_code_groups=_CODE_GROUPS,
        text_hidden_size=_HIDDEN,
        code_predictor_config=cp_config,
    )
    model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        config=cp_config,
        talker_config=talker_config,
    )
    model.eval()
    return model


@pytest.fixture()
def prefill_embeds():
    """Fixed input embeddings: [batch=1, seq=2, hidden]."""
    torch.manual_seed(42)
    return torch.randn(1, 2, _HIDDEN)


# ---------------------------------------------------------------------------
# Test A: _sample_token
# ---------------------------------------------------------------------------

class TestSampleToken:
    """Unit tests for the _sample_token utility."""

    def test_greedy_returns_argmax(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        result = _sample_token(logits, do_sample=False, top_p=1.0, top_k=0, temperature=1.0)
        assert result.shape == (1, 1)
        assert result.item() == 1  # index of 5.0

    def test_greedy_batch(self):
        logits = torch.tensor([
            [1.0, 5.0, 3.0],
            [9.0, 2.0, 3.0],
        ])
        result = _sample_token(logits, do_sample=False, top_p=1.0, top_k=0, temperature=1.0)
        assert result.shape == (2, 1)
        assert result[0].item() == 1
        assert result[1].item() == 0

    def test_output_shape(self):
        logits = torch.randn(4, 64)
        result = _sample_token(logits, do_sample=True, top_p=1.0, top_k=0, temperature=1.0)
        assert result.shape == (4, 1)

    def test_top_k_limits_vocabulary(self):
        """With top_k=1, sampling always picks the argmax."""
        torch.manual_seed(0)
        logits = torch.tensor([[1.0, 10.0, 3.0, 2.0]])
        results = set()
        for _ in range(20):
            tok = _sample_token(logits, do_sample=True, top_p=1.0, top_k=1, temperature=1.0)
            results.add(tok.item())
        assert results == {1}, f"top_k=1 should always select argmax, got {results}"

    def test_temperature_sharpens(self):
        """Very low temperature should behave like greedy."""
        torch.manual_seed(0)
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        results = set()
        for _ in range(20):
            tok = _sample_token(logits, do_sample=True, top_p=1.0, top_k=0, temperature=0.01)
            results.add(tok.item())
        assert results == {1}, f"temperature=0.01 should always select argmax, got {results}"

    def test_top_p_nucleus(self):
        """With top_p very small, only the top token should be sampled."""
        torch.manual_seed(0)
        logits = torch.tensor([[1.0, 10.0, 3.0, 2.0]])
        results = set()
        for _ in range(20):
            tok = _sample_token(logits, do_sample=True, top_p=0.01, top_k=0, temperature=1.0)
            results.add(tok.item())
        assert results == {1}, f"top_p=0.01 should always select top token, got {results}"

    def test_valid_token_range(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 64)
        tok = _sample_token(logits, do_sample=True, top_p=0.9, top_k=10, temperature=0.9)
        assert (tok >= 0).all() and (tok < 64).all()


# ---------------------------------------------------------------------------
# Test B: generate_codes render tests
# ---------------------------------------------------------------------------

class TestGenerateCodes:
    """Functional tests for generate_codes()."""

    # Canonical greedy kwargs — all sampling params are required (no defaults).
    _GREEDY = dict(do_sample=False, top_p=1.0, top_k=0, temperature=1.0)

    def test_output_shape(self, code_predictor, prefill_embeds):
        with torch.no_grad():
            result = code_predictor.generate_codes(prefill_embeds, **self._GREEDY)
        expected_codes = _CODE_GROUPS - 1  # 3
        assert result.shape == (1, expected_codes)

    def test_greedy_deterministic(self, code_predictor, prefill_embeds):
        """Greedy decoding should produce identical results across calls."""
        with torch.no_grad():
            r1 = code_predictor.generate_codes(prefill_embeds, **self._GREEDY)
            r2 = code_predictor.generate_codes(prefill_embeds, **self._GREEDY)
        assert torch.equal(r1, r2)

    def test_valid_token_range(self, code_predictor, prefill_embeds):
        with torch.no_grad():
            result = code_predictor.generate_codes(prefill_embeds, **self._GREEDY)
        assert (result >= 0).all()
        assert (result < _VOCAB).all()

    def test_batch_support(self, code_predictor):
        """generate_codes should handle batch_size > 1."""
        torch.manual_seed(42)
        embeds = torch.randn(3, 2, _HIDDEN)
        with torch.no_grad():
            result = code_predictor.generate_codes(embeds, **self._GREEDY)
        assert result.shape == (3, _CODE_GROUPS - 1)


# ---------------------------------------------------------------------------
# Test C: Regression — generate_codes() vs HF generate()
# ---------------------------------------------------------------------------

class TestGenerateCodesRegression:
    """Verify generate_codes() produces identical output to HF generate()."""

    _GREEDY = dict(do_sample=False, top_p=1.0, top_k=0, temperature=1.0)

    def test_greedy_matches_hf_generate(self, code_predictor, prefill_embeds):
        """Under greedy decoding, generate_codes() must match HF generate()."""
        with torch.no_grad():
            # New path: our manual loop
            new_result = code_predictor.generate_codes(
                prefill_embeds, **self._GREEDY,
            )

            # Old path: HF GenerationMixin.generate()
            hf_result = code_predictor.generate(
                inputs_embeds=prefill_embeds,
                max_new_tokens=_CODE_GROUPS - 1,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        hf_sequences = hf_result.sequences
        # HF generate() with inputs_embeds may prepend dummy tokens.
        # Take the last (num_code_groups - 1) tokens as the generated codes.
        num_codes = _CODE_GROUPS - 1
        hf_codes = hf_sequences[:, -num_codes:]

        assert torch.equal(new_result, hf_codes), (
            f"generate_codes() output differs from HF generate().\n"
            f"  generate_codes: {new_result}\n"
            f"  HF generate  : {hf_codes}"
        )

    def test_greedy_regression_different_inputs(self, code_predictor):
        """Regression holds for varied inputs, not just one seed."""
        for seed in [0, 7, 42, 123]:
            torch.manual_seed(seed)
            embeds = torch.randn(1, 2, _HIDDEN)

            with torch.no_grad():
                new_result = code_predictor.generate_codes(
                    embeds, **self._GREEDY,
                )

                hf_result = code_predictor.generate(
                    inputs_embeds=embeds,
                    max_new_tokens=_CODE_GROUPS - 1,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            num_codes = _CODE_GROUPS - 1
            hf_codes = hf_result.sequences[:, -num_codes:]

            assert torch.equal(new_result, hf_codes), (
                f"Regression failure at seed={seed}.\n"
                f"  generate_codes: {new_result}\n"
                f"  HF generate  : {hf_codes}"
            )
