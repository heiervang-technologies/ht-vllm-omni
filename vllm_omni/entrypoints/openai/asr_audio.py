"""Fast, bounded audio ingest for vLLM's speech-to-text endpoints."""

from __future__ import annotations

import io
import math
import os
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
import soundfile

_BAD_SF_CODES = {0, 1, 3, 4}
_DEFAULT_MAX_DECODED_AUDIO_MB = 512


class UnsafeAudioInputError(ValueError):
    """The container metadata would cause unsafe or ambiguous decoding."""


def downmix_audio(audio: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Reduce channels without NumPy's slow generic mean reduction.

    libsndfile returns channels-last and vLLM transposes it to channels-first.
    ``np.add.reduce`` is 10x+ faster than ``mean(axis=0)`` for the common
    stereo shape on amber while producing identical float32 samples.
    """
    if audio.ndim == 1:
        return np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise UnsafeAudioInputError(f"Unsupported audio shape: {audio.shape}")
    if audio.shape[0] == 2:
        # The explicit binary ufunc is substantially faster than NumPy's
        # generic reduction machinery for the overwhelmingly common case.
        mixed = np.add(audio[0], audio[1], dtype=np.float32)
        mixed *= np.float32(0.5)
    else:
        mixed = np.add.reduce(audio, axis=0, dtype=np.float32)
        mixed *= np.float32(1.0 / audio.shape[0])
    return mixed


def _max_decoded_bytes() -> int:
    raw_value = os.getenv("VLLM_OMNI_MAX_DECODED_AUDIO_MB", str(_DEFAULT_MAX_DECODED_AUDIO_MB))
    try:
        value_mb = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("VLLM_OMNI_MAX_DECODED_AUDIO_MB must be an integer") from exc
    if value_mb < 1:
        raise RuntimeError("VLLM_OMNI_MAX_DECODED_AUDIO_MB must be at least 1")
    return value_mb * 1024**2


def _validate_complete_riff(path: BinaryIO) -> None:
    """Reject a truncated classic RIFF/WAVE before invoking a codec.

    RF64 uses a sentinel size plus a ``ds64`` chunk and is left to libsndfile.
    The ASR HTTP path passes a BytesIO, so this check is allocation-free.
    """
    if not isinstance(path, io.BytesIO):
        return
    view = path.getbuffer()
    if len(view) < 12 or bytes(view[:4]) != b"RIFF" or bytes(view[8:12]) != b"WAVE":
        return
    declared_size = struct.unpack_from("<I", view, 4)[0]
    if declared_size == 0xFFFFFFFF:
        return
    declared_total = declared_size + 8
    if declared_total > len(view):
        raise UnsafeAudioInputError(
            f"Truncated WAV container: header declares {declared_total} bytes, received {len(view)}"
        )


def load_audio_soundfile(
    path: io.BytesIO | Path | str,
    *,
    sr: float | None = 22_050,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load audio with bounded decoded size and the measured mono fast path."""
    if isinstance(path, io.BytesIO):
        _validate_complete_riff(path)
        path.seek(0)

    with soundfile.SoundFile(path) as handle:
        native_sr = handle.samplerate
        decoded_bytes = handle.frames * handle.channels * np.dtype(np.float32).itemsize
        max_bytes = _max_decoded_bytes()
        if decoded_bytes > max_bytes:
            raise UnsafeAudioInputError(
                f"Decoded audio exceeds limit: {decoded_bytes / 1024**2:.1f} MiB > {max_bytes / 1024**2:.0f} MiB"
            )
        audio = handle.read(dtype="float32", always_2d=False).T

    if audio.shape[-1] == 0:
        raise UnsafeAudioInputError("Audio contains no decodable samples")
    if mono and audio.ndim > 1:
        audio = downmix_audio(audio)

    if sr is not None and not math.isclose(float(sr), float(native_sr), rel_tol=0.0, abs_tol=1e-6):
        # Keep the same libswresample implementation as vLLM 0.20.0.
        from vllm.multimodal.audio import resample_audio_pyav

        audio = resample_audio_pyav(audio, orig_sr=native_sr, target_sr=sr)
        return audio, int(sr)
    return audio, native_sr


def load_audio(
    path: io.BytesIO | Path | str,
    *,
    sr: float | None = 22_050,
    mono: bool = True,
):
    """vLLM-compatible loader with fast WAV ingest and existing PyAV fallback."""
    try:
        return load_audio_soundfile(path, sr=sr, mono=mono)
    except soundfile.LibsndfileError as exc:
        if exc.code not in _BAD_SF_CODES:
            raise

    if isinstance(path, io.BytesIO):
        path.seek(0)
    from vllm.multimodal.media.audio import load_audio_pyav

    try:
        audio, sample_rate = load_audio_pyav(path, sr=sr, mono=mono)
    except Exception as exc:
        raise ValueError("Invalid or unsupported audio file.") from exc
    if audio.nbytes > _max_decoded_bytes():
        raise UnsafeAudioInputError(
            f"Decoded audio exceeds limit: {audio.nbytes / 1024**2:.1f} MiB > "
            f"{_max_decoded_bytes() / 1024**2:.0f} MiB"
        )
    return audio, sample_rate


def install_vllm_asr_audio_loader() -> None:
    """Install the durable fork override used by vLLM's STT base class."""
    from vllm.entrypoints.openai.speech_to_text import speech_to_text

    speech_to_text.load_audio = load_audio
