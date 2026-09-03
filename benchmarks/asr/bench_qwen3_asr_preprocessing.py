#!/usr/bin/env python3
"""CPU latency and hostile-input benchmark for the Qwen3-ASR request path.

The generated corpus is deterministic and intentionally not checked in as WAV
blobs.  The JSON output records every file's SHA-256 digest so a bench run can
prove that it used the same bytes.  This script does not import vLLM or torch
and is safe to run on a serving gem with ``nice``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import struct
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import av
import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor

TARGET_SAMPLE_RATE = 16_000
HOP_LENGTH = 160


@dataclass(frozen=True)
class AudioCase:
    name: str
    duration_s: int
    sample_rate: int
    channels: int


CORPUS = (
    AudioCase("short_16k_mono", 5, 16_000, 1),
    AudioCase("clip_48k_stereo", 30, 48_000, 2),
    AudioCase("long_44k_stereo", 120, 44_100, 2),
)


def _load_asr_audio_module():
    module_path = Path(__file__).parents[2] / "vllm_omni/entrypoints/openai/asr_audio.py"
    spec = importlib.util.spec_from_file_location("_bench_asr_audio", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASR_AUDIO = _load_asr_audio_module()


def _pcm_block(case: AudioCase, start: int, count: int) -> np.ndarray:
    """Return deterministic speech-like signed PCM16 frames."""
    idx = np.arange(start, start + count, dtype=np.float64)
    t = idx / case.sample_rate
    carrier = 0.34 * np.sin(2 * np.pi * 173.0 * t)
    carrier += 0.18 * np.sin(2 * np.pi * 947.0 * t)
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.3 * t) ** 2
    mono = carrier * envelope
    # A repeatable quiet interval exercises future energy-aware chunking.
    mono[(idx.astype(np.int64) // case.sample_rate) % 11 == 10] *= 0.02
    if case.channels == 1:
        frames = mono[:, None]
    else:
        right = 0.94 * mono + 0.025 * np.sin(2 * np.pi * 311.0 * t)
        frames = np.stack((mono, right), axis=1)
    return np.round(np.clip(frames, -1.0, 1.0) * 32767).astype("<i2")


def materialize_corpus(directory: Path) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for case in CORPUS:
        path = directory / f"{case.name}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(case.channels)
            wav.setsampwidth(2)
            wav.setframerate(case.sample_rate)
            total = case.duration_s * case.sample_rate
            for start in range(0, total, case.sample_rate):
                frames = _pcm_block(case, start, min(case.sample_rate, total - start))
                wav.writeframesraw(frames.tobytes())
        payload = path.read_bytes()
        manifest.append(
            {
                **asdict(case),
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return manifest


def _resample_pyav(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Match vLLM 0.20's array -> PyAV resampling path."""
    if orig_sr == target_sr:
        return audio
    expected_len = math.ceil(audio.shape[-1] * target_sr / orig_sr)
    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) < 1024:
        audio_f32 = np.pad(audio_f32, (0, 1024 - len(audio_f32)))
    frame = av.AudioFrame.from_ndarray(audio_f32.reshape(1, -1), format="fltp", layout="mono")
    frame.sample_rate = orig_sr
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    output = resampler.resample(frame)
    output.extend(resampler.resample(None))
    result = np.concatenate([item.to_ndarray() for item in output], axis=1).squeeze(0)
    return result[:expected_len]


def _decode_soundfile(payload: bytes) -> tuple[np.ndarray, int]:
    with sf.SoundFile(io.BytesIO(payload)) as handle:
        sample_rate = handle.samplerate
        audio = handle.read(dtype="float32", always_2d=False).T
    return audio, sample_rate


def _decode_resample_fused_pyav(payload: bytes) -> np.ndarray:
    """Decode, downmix, and resample per frame instead of buffering native PCM."""
    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(payload)) as container:
        if not container.streams.audio:
            raise ValueError("No audio stream")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray())
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray())
    if not chunks:
        raise ValueError("No audio frames")
    return np.concatenate(chunks, axis=-1).astype(np.float32, copy=False).squeeze(0)


def _pad_hop(audio: np.ndarray) -> np.ndarray:
    remainder = audio.shape[-1] % HOP_LENGTH
    return audio if remainder == 0 else np.pad(audio, (0, HOP_LENGTH - remainder))


def _downmix_fast(audio: np.ndarray) -> np.ndarray:
    return ASR_AUDIO.downmix_audio(audio)


def _feature_extractor() -> WhisperFeatureExtractor:
    # Qwen/Qwen3-ASR-1.7B preprocessor_config.json, pinned explicitly so the
    # benchmark never needs model weights or a network request.
    return WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=TARGET_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=True,
    )


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _summarize(samples: list[float], duration_s: int) -> dict[str, float]:
    p50 = median(samples)
    return {
        "p50_ms": round(p50, 4),
        "p95_ms": round(_percentile(samples, 0.95), 4),
        "p50_ms_per_audio_s": round(p50 / duration_s, 6),
    }


def benchmark_case(path: Path, duration_s: int, repeats: int) -> dict[str, Any]:
    payload = path.read_bytes()
    extractor = _feature_extractor()
    timings: dict[str, list[float]] = {
        "decode": [],
        "downmix": [],
        "downmix_fast": [],
        "resample": [],
        "hop_pad": [],
        "feature_extract": [],
        "cpu_total": [],
        "cpu_total_fast_downmix": [],
        "fused_decode_resample": [],
    }
    sizes: dict[str, int] = {}

    for iteration in range(repeats + 1):
        started = time.perf_counter_ns()
        native, sample_rate = _decode_soundfile(payload)
        decoded = time.perf_counter_ns()
        mono = native.mean(axis=0, dtype=np.float32) if native.ndim > 1 else native
        downmixed = time.perf_counter_ns()
        fast_mono = _downmix_fast(native)
        fast_downmixed = time.perf_counter_ns()
        resampled = _resample_pyav(fast_mono, sample_rate, TARGET_SAMPLE_RATE)
        resample_done = time.perf_counter_ns()
        padded = _pad_hop(resampled)
        pad_done = time.perf_counter_ns()
        features = extractor(
            padded,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="np",
        )
        feature_done = time.perf_counter_ns()

        fused_started = time.perf_counter_ns()
        fused = _decode_resample_fused_pyav(payload)
        fused_done = time.perf_counter_ns()

        if iteration == 0:
            # Warm both paths and validate that the proposed fused ingest does
            # not silently change duration or gross sample values.
            if abs(len(fused) - len(resampled)) > 1:
                raise RuntimeError(f"fused length mismatch: {len(fused)} != {len(resampled)}")
            if not np.array_equal(mono, fast_mono):
                raise RuntimeError("fast downmix changed float32 samples")
            sizes = {
                "request_bytes": len(payload),
                "decoded_native_bytes": native.nbytes,
                "resampled_bytes": resampled.nbytes,
                "feature_bytes": features["input_features"].nbytes,
            }
            continue

        decode_ms = (decoded - started) / 1e6
        downmix_ms = (downmixed - decoded) / 1e6
        fast_downmix_ms = (fast_downmixed - downmixed) / 1e6
        resample_ms = (resample_done - fast_downmixed) / 1e6
        pad_ms = (pad_done - resample_done) / 1e6
        feature_ms = (feature_done - pad_done) / 1e6
        timings["decode"].append(decode_ms)
        timings["downmix"].append(downmix_ms)
        timings["downmix_fast"].append(fast_downmix_ms)
        timings["resample"].append(resample_ms)
        timings["hop_pad"].append(pad_ms)
        timings["feature_extract"].append(feature_ms)
        timings["cpu_total"].append(decode_ms + downmix_ms + resample_ms + pad_ms + feature_ms)
        timings["cpu_total_fast_downmix"].append(decode_ms + fast_downmix_ms + resample_ms + pad_ms + feature_ms)
        timings["fused_decode_resample"].append((fused_done - fused_started) / 1e6)

    return {
        "duration_s": duration_s,
        "sizes": sizes,
        "stages": {name: _summarize(values, duration_s) for name, values in timings.items()},
    }


def _hostile_inputs(valid_wav: bytes) -> dict[str, bytes]:
    lying_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0x7FFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        16_000,
        32_000,
        2,
        16,
        b"data",
        0x7FFFFFF0,
    )
    return {
        "empty": b"",
        "random_4k": hashlib.shake_256(b"qwen3-asr-hostile").digest(4096),
        "riff_only": b"RIFF\xff\xff\xff\x7fWAVE",
        "lying_wav_header": lying_header,
        "truncated_valid_wav": valid_wav[: max(45, len(valid_wav) // 5)],
    }


def _decode_worker(payload: bytes, queue: mp.Queue, optimized: bool) -> None:
    started = time.perf_counter_ns()
    try:
        if optimized:
            ASR_AUDIO.load_audio_soundfile(io.BytesIO(payload), sr=None, mono=True)
        else:
            native, sample_rate = _decode_soundfile(payload)
            mono = native.mean(axis=0, dtype=np.float32) if native.ndim > 1 else native
            _resample_pyav(mono, sample_rate, TARGET_SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001 - exception class is fuzz output
        queue.put(("rejected", type(exc).__name__, (time.perf_counter_ns() - started) / 1e6))
    else:
        queue.put(("accepted", None, (time.perf_counter_ns() - started) / 1e6))


def fuzz_ingest(valid_wav: bytes, timeout_s: float, *, optimized: bool) -> dict[str, Any]:
    results: dict[str, Any] = {}
    # ``fork`` keeps process-start/import time out of the malformed-file
    # deadline on Linux. Fall back to spawn for platforms without it.
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    context = mp.get_context(start_method)
    for name, payload in _hostile_inputs(valid_wav).items():
        queue = context.Queue()
        process = context.Process(target=_decode_worker, args=(payload, queue, optimized))
        started = time.perf_counter_ns()
        process.start()
        process.join(timeout_s)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if process.is_alive():
            process.terminate()
            process.join(1)
            results[name] = {"status": "timeout", "wall_ms": round(elapsed_ms, 4)}
        elif queue.empty():
            results[name] = {
                "status": "worker_crash",
                "exit_code": process.exitcode,
                "wall_ms": round(elapsed_ms, 4),
            }
        else:
            status, error, decode_ms = queue.get_nowait()
            results[name] = {
                "status": status,
                "error": error,
                "decode_ms": round(decode_ms, 4),
                "wall_ms": round(elapsed_ms, 4),
            }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("benchmarks/asr/.work"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hostile-timeout-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    manifest = materialize_corpus(args.work_dir)
    results: dict[str, Any] = {}
    for item in manifest:
        results[item["name"]] = benchmark_case(Path(item["path"]), item["duration_s"], args.repeats)

    report = {
        "schema_version": 1,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "nice": os.nice(0),
            "numpy": np.__version__,
            "soundfile": sf.__version__,
            "pyav": av.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "corpus": manifest,
        "cases": results,
        "hostile_inputs_baseline": fuzz_ingest(
            Path(manifest[0]["path"]).read_bytes(), args.hostile_timeout_s, optimized=False
        ),
        "hostile_inputs_optimized": fuzz_ingest(
            Path(manifest[0]["path"]).read_bytes(), args.hostile_timeout_s, optimized=True
        ),
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_stages": {
            "encoder": "could-not-measure: live worker owns amber GPU",
            "decode": "could-not-measure: live worker owns amber GPU",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
