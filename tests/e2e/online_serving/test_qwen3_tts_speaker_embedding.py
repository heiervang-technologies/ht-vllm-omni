# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for speaker_embedding API parameter.

Tests both model types:
- 0.6B-Base: supports Base task with speaker_embedding (expected to work)
- 0.6B-CustomVoice: does NOT support Base task (expected 400 error)

The speaker_embedding parameter allows direct voice cloning by injecting a
pre-computed ECAPA-TDNN embedding, bypassing ref_audio extraction.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import struct
from pathlib import Path

import httpx
import pytest

from tests.conftest import OmniServer
from tests.utils import hardware_test

MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_CUSTOMVOICE = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# A synthetic 1024-dim speaker embedding (all 0.1 — not a real voice, but
# exercises the full code path through the talker's _build_prompt_embeds).
DUMMY_EMBEDDING_1024 = [0.1] * 1024

SYN_TEXT = "Hello."
MIN_AUDIO_BYTES = 2000
# Limit generation to keep tests fast (dummy embeddings produce nonsensical
# output that may never hit a natural stop token).
MAX_NEW_TOKENS = 256


def get_stage_config():
    return str(
        Path(__file__).parent.parent.parent.parent
        / "vllm_omni"
        / "model_executor"
        / "stage_configs"
        / "qwen3_tts.yaml"
    )


def _server_args():
    return [
        "--stage-configs-path",
        get_stage_config(),
        "--stage-init-timeout",
        "120",
        "--trust-remote-code",
        "--enforce-eager",
        "--disable-log-stats",
    ]


def verify_wav_audio(content: bytes) -> bool:
    if len(content) < 44:
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


def assert_not_silence(pcm_bytes: bytes):
    """Assert PCM16 samples are not all identical (e.g. all-silence)."""
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    unique = set(samples)
    assert len(unique) > 1, f"All-silence detected: {len(samples)} samples, unique values: {unique}"


# ── 0.6B-Base model tests ──


@pytest.fixture(scope="module")
def base_server():
    """Start vLLM-Omni server with 0.6B-Base model."""
    with OmniServer(MODEL_BASE, _server_args()) as server:
        yield server


class TestSpeakerEmbeddingBase:
    """Speaker embedding tests against the 0.6B-Base model (supports Base task)."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_produces_audio(self, base_server) -> None:
        """speaker_embedding with Base task produces valid WAV audio."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "x_vector_only_mode": True,
            "response_format": "wav",
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio too small: {len(response.content)} bytes"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_pcm_not_silence(self, base_server) -> None:
        """speaker_embedding PCM output contains real audio, not all-silence."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "x_vector_only_mode": True,
            "response_format": "pcm",
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert len(response.content) > MIN_AUDIO_BYTES
        assert_not_silence(response.content)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_streaming(self, base_server) -> None:
        """speaker_embedding works with streaming PCM output."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "x_vector_only_mode": True,
            "response_format": "pcm",
            "stream": True,
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert "audio/pcm" in response.headers.get("content-type", "")
        assert len(response.content) > MIN_AUDIO_BYTES
        assert_not_silence(response.content)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_mutually_exclusive_with_ref_audio(self, base_server) -> None:
        """Sending both speaker_embedding and ref_audio returns 400."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "ref_audio": "https://example.com/audio.wav",
            "response_format": "wav",
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
        assert "mutually exclusive" in response.text

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_empty_rejected(self, base_server) -> None:
        """Empty speaker_embedding list returns 400."""
        url = f"http://{base_server.host}:{base_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_BASE,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": [],
            "response_format": "wav",
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
        assert "non-empty" in response.text


# ── 0.6B-CustomVoice model tests ──


@pytest.fixture(scope="module")
def customvoice_server():
    """Start vLLM-Omni server with 0.6B-CustomVoice model."""
    with OmniServer(MODEL_CUSTOMVOICE, _server_args()) as server:
        yield server


class TestSpeakerEmbeddingCustomVoice:
    """Speaker embedding tests against 0.6B-CustomVoice model.

    The CustomVoice model also supports Base task with speaker_embedding.
    An earlier bug (tensor lost during msgspec IPC serialization) made
    this appear broken — it now works correctly.
    """

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speaker_embedding_works_on_customvoice_model(self, customvoice_server) -> None:
        """speaker_embedding with Base task works even on CustomVoice model."""
        url = f"http://{customvoice_server.host}:{customvoice_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_CUSTOMVOICE,
            "input": SYN_TEXT,
            "task_type": "Base",
            "speaker_embedding": DUMMY_EMBEDDING_1024,
            "x_vector_only_mode": True,
            "response_format": "wav",
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert verify_wav_audio(response.content), "Response is not valid WAV"
        assert len(response.content) > MIN_AUDIO_BYTES

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_customvoice_normal_voice_still_works(self, customvoice_server) -> None:
        """Verify the CustomVoice model still works normally with named voices."""
        url = f"http://{customvoice_server.host}:{customvoice_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL_CUSTOMVOICE,
            "input": "Hello, this is a normal voice test.",
            "voice": "vivian",
            "language": "English",
            "response_format": "wav",
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert verify_wav_audio(response.content), "Response is not valid WAV"
        assert len(response.content) > MIN_AUDIO_BYTES
