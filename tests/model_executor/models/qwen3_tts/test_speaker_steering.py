"""Unit tests for Qwen3-TTS dynamic speaker steering (speaker_embedding_end)."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts import qwen3_tts_talker
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

H = 8
Talker = Qwen3TTSTalkerForConditionalGeneration


def _prepare(info: dict, start: torch.Tensor, num_text_tokens: int = 10) -> dict:
    Talker._prepare_speaker_steering(None, info, start.view(1, 1, -1), num_text_tokens=num_text_tokens)
    return info


def test_prepare_noop_without_end():
    info = _prepare({}, torch.zeros(H))
    assert "_steering_delta" not in info


def test_prepare_delta_and_explicit_frames():
    start = torch.full((H,), 1.0)
    info = _prepare({"speaker_embedding_end": [[3.0] * H], "speaker_steering_frames": [24]}, start)
    assert torch.allclose(info["_steering_delta"], torch.full((H,), 2.0))
    assert info["_steering_frames"] == 24


def test_prepare_default_frames_scale_with_text():
    short = _prepare({"speaker_embedding_end": [[0.0] * H]}, torch.ones(H), num_text_tokens=1)
    long = _prepare({"speaker_embedding_end": [[0.0] * H]}, torch.ones(H), num_text_tokens=100)
    assert short["_steering_frames"] == qwen3_tts_talker._STEERING_MIN_FRAMES
    assert long["_steering_frames"] == int(100 * qwen3_tts_talker._STEERING_FRAMES_PER_TEXT_TOKEN)


def test_prepare_dim_mismatch_raises():
    with pytest.raises(ValueError, match="expected 8"):
        _prepare({"speaker_embedding_end": [[0.0] * (H + 1)]}, torch.ones(H))


def _fake_talker():
    return SimpleNamespace(
        _model_dtype=lambda: torch.float32,
        embed_input_ids=lambda ids: torch.zeros(*ids.shape, H),
    )


def _decode(meta: dict, embed_extra: dict) -> tuple[torch.Tensor, dict]:
    info = {
        "text": ["hi"],
        "task_type": ["Base"],
        "embed": {"tts_pad": torch.zeros(1, H), **embed_extra},
        "hidden_states": {"trailing_text": torch.ones(3, H), "last": torch.zeros(1, H)},
        "meta": meta,
    }
    _, _, update = Talker.preprocess(_fake_talker(), torch.tensor([5]), None, **info)
    _, text_step = update["mtp_inputs"]
    return text_step, update


def test_decode_without_steering_is_unchanged():
    text_step, update = _decode({}, {})
    assert torch.equal(text_step, torch.ones(1, H))
    assert "steering_step" not in update["meta"]


@pytest.mark.parametrize("step,expected_ramp", [(0, 0.0), (5, 0.5), (10, 1.0), (40, 1.0)])
def test_decode_ramps_delta_into_text_step(step, expected_ramp):
    delta = torch.full((H,), 2.0)
    text_step, update = _decode({"steering_step": step, "steering_frames": 10}, {"steering_delta": delta})
    assert torch.allclose(text_step, torch.ones(1, H) + expected_ramp * delta)
    assert update["meta"]["steering_step"] == step + 1
