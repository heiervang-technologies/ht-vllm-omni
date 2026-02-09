# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for streaming talker loop vs HF generate() equivalence.

Verifies that the manual autoregressive loop (used by generate_streaming())
produces identical codec codes to HF's generate() under greedy decoding.

Uses the same compile+exec test pattern as test_code_predictor_generate.py
to bypass the heavy vllm_omni.__init__ import chain.
"""

import logging
import sys
import types
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Bootstrap: stub vllm-specific modules that the modeling code imports
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[4]
_saved: dict[str, types.ModuleType | None] = {}

_STUB_FQNS = [
    "vllm",
    "vllm.config",
    "vllm.logger",
    "vllm.sequence",
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

    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)
    sys.modules["vllm.config"].VllmConfig = type("VllmConfig", (), {})

    weight_mod = sys.modules["vllm_omni.model_executor.model_loader.weight_utils"]
    weight_mod.download_weights_from_hf_specific = lambda *a, **kw: None

    sys.modules["vllm_omni.model_executor.models.qwen3_omni"].Qwen3OmniMoeForConditionalGeneration = type(
        "Stub", (), {}
    )
    sys.modules["vllm_omni.model_executor.models.registry"].OmniModelRegistry = type("Stub", (), {})

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

    for fqn in _STUB_FQNS:
        parts = fqn.split(".")
        if len(parts) > 1:
            parent = sys.modules.get(".".join(parts[:-1]))
            child = sys.modules.get(fqn)
            if parent and child:
                setattr(parent, parts[-1], child)


_setup_stubs()

# Now import the real modeling code
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (  # noqa: E402
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.modeling_qwen3_tts import (  # noqa: E402
    Qwen3TTSTalkerForConditionalGeneration,
    _apply_repetition_penalty,
    _sample_token,
)


# ---------------------------------------------------------------------------
# Small model dimensions for fast tests
# ---------------------------------------------------------------------------
_VOCAB = 64
_HIDDEN = 32
_INTERMEDIATE = 64
_HEADS = 4
_KV_HEADS = 2
_LAYERS = 1
_CODE_GROUPS = 4
_TEXT_HIDDEN = 32
_EOS_ID = 63  # Use last token as EOS
_PAD_ID = 60
_BOS_ID = 61


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _TalkerModelWrapper:
    """Minimal wrapper that provides .talker and .config.talker_config."""

    def __init__(self, talker, talker_config):
        self.talker = talker
        self.config = types.SimpleNamespace(talker_config=talker_config)

    def eval(self):
        self.talker.eval()
        return self


@pytest.fixture()
def talker_model():
    """Build a tiny Qwen3TTSTalkerForConditionalGeneration with random weights."""
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
    # head_dim = hidden / heads = 32 / 4 = 8; mrope_section sums to head_dim / 2 = 4
    _rope_scaling = {
        "rope_type": "default",
        "mrope_section": [2, 1, 1],
        "interleaved": False,
    }
    talker_config = Qwen3TTSTalkerConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        num_code_groups=_CODE_GROUPS,
        text_hidden_size=_TEXT_HIDDEN,
        code_predictor_config=cp_config,
        codec_eos_token_id=_EOS_ID,
        codec_pad_id=_PAD_ID,
        codec_bos_id=_BOS_ID,
        spk_id={"test_speaker": 0},
        codec_language_id={"auto": 0},
        max_position_embeddings=128,
        pad_token_id=0,
        rope_scaling=_rope_scaling,
        # text_vocab_size is needed by Qwen3TTSTalkerModel for text_embedding
        text_vocab_size=_VOCAB,
    )

    # Transformers 5.x maps rope_scaling → rope_parameters via attribute_map.
    # Patch the config CLASS so any newly-constructed instance has the fallback.
    _patched = False
    if not hasattr(Qwen3TTSTalkerConfig, "rope_parameters"):
        Qwen3TTSTalkerConfig.rope_parameters = None
        _patched = True

    try:
        talker = Qwen3TTSTalkerForConditionalGeneration(talker_config)
    finally:
        if _patched:
            del Qwen3TTSTalkerConfig.rope_parameters

    wrapper = _TalkerModelWrapper(talker, talker_config)
    wrapper.eval()
    return wrapper


@pytest.fixture()
def prepared_inputs(talker_model):
    """Create prepared talker inputs for testing.

    Returns the same inputs that _prepare_talker_inputs would produce,
    but constructed directly to avoid needing a processor/tokenizer.
    """
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 4
    hidden = _HIDDEN

    talker_input_embeds = torch.randn(batch_size, seq_len, hidden)
    talker_attention_mask = torch.ones(batch_size, seq_len)
    trailing_text_hiddens = torch.randn(batch_size, 10, hidden)
    tts_pad_embed = torch.randn(1, 1, hidden)

    return talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed


# ---------------------------------------------------------------------------
# Test: manual streaming loop vs HF generate() equivalence
# ---------------------------------------------------------------------------

class TestStreamingGenerateEquivalence:
    """Verify the manual streaming loop produces identical codec codes to
    HF's generate() under greedy decoding."""

    _GREEDY = dict(
        do_sample=False,
        top_p=1.0,
        top_k=0,
        temperature=1.0,
        repetition_penalty=1.0,
        subtalker_dosample=False,
        subtalker_top_k=0,
        subtalker_top_p=1.0,
        subtalker_temperature=1.0,
        max_new_tokens=20,
    )

    def _run_generate(self, model, inputs):
        """Run HF generate() on the talker and return codec codes."""
        talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed = inputs
        eos_id = model.config.talker_config.codec_eos_token_id

        talker_kwargs = {
            "max_new_tokens": self._GREEDY["max_new_tokens"],
            "min_new_tokens": 2,
            "do_sample": self._GREEDY["do_sample"],
            "top_k": self._GREEDY["top_k"],
            "top_p": self._GREEDY["top_p"],
            "temperature": self._GREEDY["temperature"],
            "subtalker_dosample": self._GREEDY["subtalker_dosample"],
            "subtalker_top_k": self._GREEDY["subtalker_top_k"],
            "subtalker_top_p": self._GREEDY["subtalker_top_p"],
            "subtalker_temperature": self._GREEDY["subtalker_temperature"],
            "eos_token_id": eos_id,
            "repetition_penalty": self._GREEDY["repetition_penalty"],
            # No suppress_tokens for tiny test vocab (only relevant for real models)
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }

        talker_result = model.talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            **talker_kwargs,
        )

        talker_codes = torch.stack(
            [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None],
            dim=1,
        )

        # Trim at EOS
        first_codebook = talker_codes[:, :, 0]
        is_stop = first_codebook == eos_id
        stop_idx = torch.argmax(is_stop.int(), dim=1)
        has_stop = is_stop.any(dim=1)
        eff_len = torch.where(has_stop, stop_idx, talker_codes.shape[1])

        return [talker_codes[i, :length] for i, length in enumerate(eff_len)]

    def test_streaming_codec_codes_match_greedy(self, talker_model, prepared_inputs):
        """Under greedy decoding, the manual streaming loop and HF generate()
        must produce identical first-codebook token sequences."""
        with torch.no_grad():
            # Non-streaming path (HF generate)
            ref_codes_list = self._run_generate(talker_model, prepared_inputs)
            ref_codes = ref_codes_list[0]  # [T, num_codebooks]

            # Streaming path (manual loop)
            talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed = prepared_inputs

            streaming_chunks = []
            for codes_chunk, hidden_chunk, is_final in _run_manual_streaming_loop(
                talker_model, talker_input_embeds, talker_attention_mask,
                trailing_text_hiddens, tts_pad_embed, self._GREEDY,
            ):
                if codes_chunk is not None:
                    streaming_chunks.append(codes_chunk)
                if is_final:
                    break

        if streaming_chunks:
            streaming_codes = torch.cat(streaming_chunks, dim=1)[0]  # [T, num_codebooks]
        else:
            streaming_codes = torch.empty(0, _CODE_GROUPS)

        assert ref_codes.shape == streaming_codes.shape, (
            f"Shape mismatch: generate={ref_codes.shape}, streaming={streaming_codes.shape}"
        )
        assert torch.equal(ref_codes, streaming_codes), (
            f"Codec codes differ.\n"
            f"  generate()  first-codebook: {ref_codes[:, 0].tolist()}\n"
            f"  streaming() first-codebook: {streaming_codes[:, 0].tolist()}"
        )

    @pytest.mark.parametrize("chunk_size", [3, 5, 10, 50])
    def test_chunk_size_does_not_affect_output(self, talker_model, prepared_inputs, chunk_size):
        """Different chunk sizes must produce identical codec codes."""
        with torch.no_grad():
            talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed = prepared_inputs

            chunks = []
            for codes_chunk, _, is_final in _run_manual_streaming_loop(
                talker_model, talker_input_embeds, talker_attention_mask,
                trailing_text_hiddens, tts_pad_embed,
                {**self._GREEDY, "chunk_size": chunk_size},
            ):
                if codes_chunk is not None:
                    chunks.append(codes_chunk)
                if is_final:
                    break

        if chunks:
            codes = torch.cat(chunks, dim=1)[0]
        else:
            codes = torch.empty(0, _CODE_GROUPS)

        # Compare against chunk_size=50 (effectively no chunking for short sequences)
        with torch.no_grad():
            ref_chunks = []
            for codes_chunk, _, is_final in _run_manual_streaming_loop(
                talker_model, talker_input_embeds, talker_attention_mask,
                trailing_text_hiddens, tts_pad_embed,
                {**self._GREEDY, "chunk_size": 50},
            ):
                if codes_chunk is not None:
                    ref_chunks.append(codes_chunk)
                if is_final:
                    break

        if ref_chunks:
            ref_codes = torch.cat(ref_chunks, dim=1)[0]
        else:
            ref_codes = torch.empty(0, _CODE_GROUPS)

        assert torch.equal(codes, ref_codes), (
            f"chunk_size={chunk_size} produced different codes.\n"
            f"  ref: {ref_codes[:, 0].tolist()}\n"
            f"  got: {codes[:, 0].tolist()}"
        )


def _run_manual_streaming_loop(model, talker_input_embeds, talker_attention_mask,
                                trailing_text_hiddens, tts_pad_embed, gen_kwargs):
    """Run the manual streaming talker loop (mirrors generate_streaming internals).

    This directly exercises the same manual prefill + decode loop from
    generate_streaming(), using pre-prepared talker inputs to bypass
    the full _prepare_talker_inputs() pipeline.
    """
    eos_id = model.config.talker_config.codec_eos_token_id
    chunk_size = gen_kwargs.get("chunk_size", 50)
    max_new_tokens = gen_kwargs.get("max_new_tokens", 20)
    do_sample = gen_kwargs.get("do_sample", False)
    top_p = gen_kwargs.get("top_p", 1.0)
    top_k = gen_kwargs.get("top_k", 0)
    temperature = gen_kwargs.get("temperature", 1.0)
    repetition_penalty = gen_kwargs.get("repetition_penalty", 1.0)
    subtalker_dosample = gen_kwargs.get("subtalker_dosample", False)
    subtalker_top_p = gen_kwargs.get("subtalker_top_p", 1.0)
    subtalker_top_k = gen_kwargs.get("subtalker_top_k", 0)
    subtalker_temperature = gen_kwargs.get("subtalker_temperature", 1.0)

    # No suppress_tokens for tiny test vocab (only relevant for real models)
    suppress_token_ids = torch.tensor([], dtype=torch.long)

    device = talker_input_embeds.device
    batch_size = talker_input_embeds.shape[0]
    prefill_len = talker_input_embeds.shape[1]

    # Prefill
    outputs = model.talker(
        inputs_embeds=talker_input_embeds,
        attention_mask=talker_attention_mask,
        use_cache=True,
        output_hidden_states=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        subtalker_dosample=subtalker_dosample,
        subtalker_top_p=subtalker_top_p,
        subtalker_top_k=subtalker_top_k,
        subtalker_temperature=subtalker_temperature,
    )

    logits = outputs.logits[:, -1, :]
    logits[:, suppress_token_ids] = float("-inf")
    next_token = _sample_token(logits, do_sample, top_p, top_k, temperature)

    past_hidden = outputs.past_hidden
    generation_step = outputs.generation_step
    past_kv = outputs.past_key_values
    current_pos = prefill_len

    all_codec_ids = []
    all_hidden_states = []
    generated_ids = [next_token.squeeze(-1)]

    # The prefill already produced the first new token, so we have
    # (max_new_tokens - 1) decode steps remaining to match HF generate.
    for _ in range(max_new_tokens - 1):
        if batch_size == 1 and next_token.squeeze().item() == eos_id:
            break

        talker_attention_mask = torch.cat(
            [talker_attention_mask,
             torch.ones(batch_size, 1, device=device, dtype=talker_attention_mask.dtype)],
            dim=1,
        )
        cache_position = torch.tensor([current_pos], device=device, dtype=torch.long)

        outputs = model.talker(
            input_ids=next_token,
            attention_mask=talker_attention_mask,
            cache_position=cache_position,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            past_hidden=past_hidden,
            generation_step=generation_step,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_p=subtalker_top_p,
            subtalker_top_k=subtalker_top_k,
            subtalker_temperature=subtalker_temperature,
        )

        codec_ids = outputs.hidden_states[1]
        if codec_ids is not None:
            all_codec_ids.append(codec_ids)
            all_hidden_states.append(outputs.past_hidden)

        past_hidden = outputs.past_hidden
        generation_step = outputs.generation_step
        past_kv = outputs.past_key_values
        current_pos += 1

        logits = outputs.logits[:, -1, :]
        logits[:, suppress_token_ids] = float("-inf")
        if repetition_penalty != 1.0:
            _apply_repetition_penalty(logits, generated_ids, repetition_penalty)
        next_token = _sample_token(logits, do_sample, top_p, top_k, temperature)
        generated_ids.append(next_token.squeeze(-1))

        if len(all_codec_ids) >= chunk_size:
            codes_chunk = torch.stack(all_codec_ids, dim=1)
            hidden_chunk = torch.cat(all_hidden_states, dim=1)
            yield codes_chunk, hidden_chunk, False
            all_codec_ids = []
            all_hidden_states = []

    if all_codec_ids:
        codes_chunk = torch.stack(all_codec_ids, dim=1)
        hidden_chunk = torch.cat(all_hidden_states, dim=1)
        yield codes_chunk, hidden_chunk, True
    else:
        yield None, None, True
