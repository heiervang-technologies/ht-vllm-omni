#!/usr/bin/env python3
"""Benchmark the upstream realtime ASR shift buffer against a ring buffer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from statistics import median
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 5


class AudioBuffer(Protocol):
    def write_audio(self, audio: np.ndarray) -> None: ...

    def read_audio(self) -> np.ndarray | None: ...

    def flush(self) -> np.ndarray | None: ...


class UpstreamShiftBuffer:
    """vLLM 0.20.0 Qwen3ASRRealtimeBuffer, copied for the baseline."""

    def __init__(self, sampling_rate: int, segment_duration_s: float = 5.0):
        self._segment_size = int(segment_duration_s * sampling_rate)
        self._buffer_size = 60 * sampling_rate
        self._buffer = np.empty(self._buffer_size, dtype=np.float32)
        self._filled_len = 0

    def write_audio(self, audio: np.ndarray) -> None:
        put_end = self._filled_len + len(audio)
        if put_end > self._buffer_size:
            new_size = max(self._buffer_size * 2, put_end)
            new_buffer = np.empty(new_size, dtype=np.float32)
            new_buffer[: self._filled_len] = self._buffer[: self._filled_len]
            self._buffer = new_buffer
            self._buffer_size = new_size
        self._buffer[self._filled_len : put_end] = audio
        self._filled_len = put_end

    def read_audio(self) -> np.ndarray | None:
        if self._filled_len < self._segment_size:
            return None
        segment = self._buffer[: self._segment_size].copy()
        remaining = self._filled_len - self._segment_size
        if remaining > 0:
            self._buffer[:remaining] = self._buffer[self._segment_size : self._filled_len]
        self._filled_len = remaining
        return segment

    def flush(self) -> np.ndarray | None:
        if self._filled_len == 0:
            return None
        audio = self._buffer[: self._filled_len].copy()
        self._filled_len = 0
        return audio


class CandidateRingBuffer:
    """Candidate implementation; production code is added only after baseline."""

    def __init__(self, sampling_rate: int, segment_duration_s: float = 5.0):
        self._segment_size = int(segment_duration_s * sampling_rate)
        self._buffer = np.empty(self._segment_size, dtype=np.float32)
        self._head = 0
        self._tail = 0
        self._filled_len = 0

    def write_audio(self, audio: np.ndarray) -> None:
        chunk = np.ascontiguousarray(audio, dtype=np.float32)
        required = self._filled_len + len(chunk)
        if required > len(self._buffer):
            new_buffer = np.empty(max(len(self._buffer) * 2, required), dtype=np.float32)
            first = min(self._filled_len, len(self._buffer) - self._head)
            new_buffer[:first] = self._buffer[self._head : self._head + first]
            new_buffer[first : self._filled_len] = self._buffer[: self._filled_len - first]
            self._buffer = new_buffer
            self._head = 0
            self._tail = self._filled_len
        first = min(len(chunk), len(self._buffer) - self._tail)
        self._buffer[self._tail : self._tail + first] = chunk[:first]
        if first < len(chunk):
            self._buffer[: len(chunk) - first] = chunk[first:]
        self._tail += len(chunk)
        if self._tail >= len(self._buffer):
            self._tail -= len(self._buffer)
        self._filled_len += len(chunk)

    def _take(self, size: int) -> np.ndarray:
        output = np.empty(size, dtype=np.float32)
        first = min(size, len(self._buffer) - self._head)
        output[:first] = self._buffer[self._head : self._head + first]
        if first < size:
            output[first:] = self._buffer[: size - first]
        self._head += size
        if self._head >= len(self._buffer):
            self._head -= len(self._buffer)
        self._filled_len -= size
        return output

    def read_audio(self) -> np.ndarray | None:
        return self._take(self._segment_size) if self._filled_len >= self._segment_size else None

    def flush(self) -> np.ndarray | None:
        return self._take(self._filled_len) if self._filled_len else None


def _audio(duration_s: int) -> np.ndarray:
    idx = np.arange(duration_s * SAMPLE_RATE, dtype=np.uint32)
    return ((idx % 65_521).astype(np.float32) / 32_760.5) - 1.0


def _exercise(buffer: AudioBuffer, audio: np.ndarray, write_samples: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(audio), write_samples):
        buffer.write_audio(audio[start : start + write_samples])
        while (segment := buffer.read_audio()) is not None:
            digest.update(segment.tobytes())
    tail = buffer.flush()
    if tail is not None:
        digest.update(tail.tobytes())
    return digest.hexdigest()


def _measure(cls: type[AudioBuffer], audio: np.ndarray, write_samples: int, repeats: int) -> dict:
    samples: list[float] = []
    expected_hash = hashlib.sha256(audio.tobytes()).hexdigest()
    for iteration in range(repeats + 1):
        started = time.perf_counter_ns()
        actual_hash = _exercise(cls(SAMPLE_RATE, SEGMENT_SECONDS), audio, write_samples)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if actual_hash != expected_hash:
            raise RuntimeError(f"buffer changed audio: {actual_hash} != {expected_hash}")
        if iteration:
            samples.append(elapsed_ms)
    return {"p50_ms": round(median(samples), 4), "samples_ms": [round(value, 4) for value in samples]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    workloads = {
        "realtime_10min_20ms_writes": (600, SAMPLE_RATE // 50),
        "burst_60s_single_write": (60, 60 * SAMPLE_RATE),
        "burst_10min_single_write": (600, 600 * SAMPLE_RATE),
    }
    results = {}
    for name, (duration_s, write_samples) in workloads.items():
        audio = _audio(duration_s)
        baseline = _measure(UpstreamShiftBuffer, audio, write_samples, args.repeats)
        candidate = _measure(CandidateRingBuffer, audio, write_samples, args.repeats)
        baseline_p50 = baseline["p50_ms"]
        candidate_p50 = candidate["p50_ms"]
        results[name] = {
            "duration_s": duration_s,
            "write_samples": write_samples,
            "upstream_shift": baseline,
            "candidate_ring": candidate,
            "p50_change_percent": round((candidate_p50 / baseline_p50 - 1) * 100, 2),
        }
    report = {
        "schema_version": 1,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "nice": os.nice(0),
        },
        "idle_preallocation_bytes": {
            "upstream_shift": 60 * SAMPLE_RATE * np.dtype(np.float32).itemsize,
            "candidate_ring": SEGMENT_SECONDS * SAMPLE_RATE * np.dtype(np.float32).itemsize,
        },
        "workloads": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
