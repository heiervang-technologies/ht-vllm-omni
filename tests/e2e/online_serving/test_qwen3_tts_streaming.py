# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for TTS HTTP streaming (/v1/audio/speech with stream=True).

Verifies that streaming PCM audio works against a real vLLM-Omni server
with the CustomVoice model (the primary streaming use case).
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

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# Short text to keep generation fast.
SYN_TEXT = "Hello, streaming test."
MIN_AUDIO_BYTES = 2000


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


@pytest.fixture(scope="module")
def streaming_server():
    """Start vLLM-Omni server with CustomVoice model."""
    with OmniServer(MODEL, _server_args()) as server:
        yield server


def assert_not_silence(pcm_bytes: bytes):
    """Assert PCM16 samples are not all identical."""
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    unique = set(samples)
    assert len(unique) > 1, f"All-silence: {len(samples)} samples, unique={unique}"


class TestStreamingCustomVoice:
    """E2E streaming tests against the CustomVoice model."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_pcm_produces_audio(self, streaming_server) -> None:
        """stream=True with PCM format returns non-empty audio/pcm response."""
        url = f"http://{streaming_server.host}:{streaming_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL,
            "input": SYN_TEXT,
            "voice": "vivian",
            "language": "English",
            "stream": True,
            "response_format": "pcm",
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert "audio/pcm" in response.headers.get("content-type", "")
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio too small: {len(response.content)} bytes"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_pcm_not_silence(self, streaming_server) -> None:
        """Streaming PCM output contains real audio, not all-silence."""
        url = f"http://{streaming_server.host}:{streaming_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL,
            "input": SYN_TEXT,
            "voice": "vivian",
            "language": "English",
            "stream": True,
            "response_format": "pcm",
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert len(response.content) > MIN_AUDIO_BYTES
        assert_not_silence(response.content)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_receives_chunked_transfer(self, streaming_server) -> None:
        """Streaming response uses chunked transfer encoding (no content-length)."""
        url = f"http://{streaming_server.host}:{streaming_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL,
            "input": SYN_TEXT,
            "voice": "vivian",
            "language": "English",
            "stream": True,
            "response_format": "pcm",
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200
        # Streaming responses should NOT have content-length (chunked transfer).
        assert "content-length" not in response.headers, (
            "Streaming response should use chunked transfer, not content-length"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_wav_rejected(self, streaming_server) -> None:
        """stream=True with response_format=wav returns 422 (protocol validation)."""
        url = f"http://{streaming_server.host}:{streaming_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL,
            "input": SYN_TEXT,
            "voice": "vivian",
            "stream": True,
            "response_format": "wav",
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 422, (
            f"Expected 422 for stream+wav, got {response.status_code}: {response.text}"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_with_speed_rejected(self, streaming_server) -> None:
        """stream=True with speed!=1.0 returns 422."""
        url = f"http://{streaming_server.host}:{streaming_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL,
            "input": SYN_TEXT,
            "voice": "vivian",
            "stream": True,
            "response_format": "pcm",
            "speed": 1.5,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 422, (
            f"Expected 422 for stream+speed, got {response.status_code}: {response.text}"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_non_streaming_still_works(self, streaming_server) -> None:
        """Non-streaming WAV request still works on the same server."""
        url = f"http://{streaming_server.host}:{streaming_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL,
            "input": SYN_TEXT,
            "voice": "vivian",
            "language": "English",
            "response_format": "wav",
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert response.content[:4] == b"RIFF"
        assert len(response.content) > MIN_AUDIO_BYTES
