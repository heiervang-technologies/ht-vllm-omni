import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")


def _load_module():
    source = Path(__file__).parents[3] / "vllm_omni/entrypoints/openai/asr_audio.py"
    spec = importlib.util.spec_from_file_location("_test_asr_audio", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


asr_audio = _load_module()


def _wav_bytes(audio: np.ndarray, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_downmix_fast_path_is_bit_identical_to_float32_mean() -> None:
    rng = np.random.default_rng(1234)
    channels_first = rng.uniform(-1, 1, size=(2, 48_000)).astype(np.float32)

    expected = channels_first.mean(axis=0, dtype=np.float32)
    actual = asr_audio.downmix_audio(channels_first)

    assert np.array_equal(actual, expected)


def test_soundfile_loader_rejects_truncated_wav() -> None:
    payload = _wav_bytes(np.zeros(16_000, dtype=np.float32))
    truncated = payload[: len(payload) // 2]

    with pytest.raises(asr_audio.UnsafeAudioInputError, match="Truncated WAV"):
        asr_audio.load_audio_soundfile(io.BytesIO(truncated), sr=None)


def test_soundfile_loader_rejects_empty_wav() -> None:
    payload = _wav_bytes(np.empty(0, dtype=np.float32))

    with pytest.raises(asr_audio.UnsafeAudioInputError, match="no decodable samples"):
        asr_audio.load_audio_soundfile(io.BytesIO(payload), sr=None)


def test_soundfile_loader_bounds_decoded_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _wav_bytes(np.zeros(300_000, dtype=np.float32))
    monkeypatch.setenv("VLLM_OMNI_MAX_DECODED_AUDIO_MB", "1")

    with pytest.raises(asr_audio.UnsafeAudioInputError, match="Decoded audio exceeds limit"):
        asr_audio.load_audio_soundfile(io.BytesIO(payload), sr=None)


def test_soundfile_loader_downmixes_stereo() -> None:
    left = np.linspace(-0.8, 0.8, 16_000, dtype=np.float32)
    right = np.linspace(0.5, -0.5, 16_000, dtype=np.float32)
    payload = _wav_bytes(np.stack((left, right), axis=1))

    actual, sample_rate = asr_audio.load_audio_soundfile(io.BytesIO(payload), sr=None)

    with sf.SoundFile(io.BytesIO(payload)) as handle:
        decoded = handle.read(dtype="float32", always_2d=False).T
    assert sample_rate == 16_000
    assert np.array_equal(actual, decoded.mean(axis=0, dtype=np.float32))
