#!/usr/bin/env python3
"""Concurrent latency, WER/CER, and power benchmark for Qwen3-ASR serving."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import threading
import time
import unicodedata
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class Item:
    id: str
    audio_path: Path
    reference: str
    language: str | None


class PowerSampler:
    def __init__(self) -> None:
        self.samples: list[tuple[float, float, float, float]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            try:
                power, memory, utilization = (float(part.strip()) for part in line.split(","))
            except ValueError:
                continue
            self.samples.append((time.perf_counter(), power, memory, utilization))

    def stop(self) -> dict[str, float | int | None]:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=2)
        joules = 0.0
        for left, right in zip(self.samples, self.samples[1:]):
            joules += (left[1] + right[1]) * 0.5 * (right[0] - left[0])
        return {
            "samples": len(self.samples),
            "joules": round(joules, 4),
            "mean_watts": round(mean(sample[1] for sample in self.samples), 4) if self.samples else None,
            "peak_memory_mib": round(max(sample[2] for sample in self.samples), 2) if self.samples else None,
            "mean_gpu_utilization_percent": (
                round(mean(sample[3] for sample in self.samples), 2) if self.samples else None
            ),
        }


def _multipart(item: Item, model: str | None) -> tuple[bytes, str]:
    boundary = f"----qwen3asr{uuid.uuid4().hex}"
    pieces: list[bytes] = []

    def field(name: str, value: str) -> None:
        pieces.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )

    field("stream", "true")
    field("stream_include_usage", "true")
    if model:
        field("model", model)
    if item.language:
        field("language", item.language)
    payload = item.audio_path.read_bytes()
    pieces.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{item.audio_path.name}"\r\n'.encode(),
            b"Content-Type: audio/wav\r\n\r\n",
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(pieces), boundary


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
    return " ".join(text.split())


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def _error_rate(reference: str, hypothesis: str, language: str | None) -> tuple[str, int, int]:
    reference_norm = _normalize(reference)
    hypothesis_norm = _normalize(hypothesis)
    if language and language.startswith("zh"):
        metric = "cer"
        reference_units = list(reference_norm.replace(" ", ""))
        hypothesis_units = list(hypothesis_norm.replace(" ", ""))
    else:
        metric = "wer"
        reference_units = reference_norm.split()
        hypothesis_units = hypothesis_norm.split()
    return metric, _edit_distance(reference_units, hypothesis_units), len(reference_units)


def _request(item: Item, endpoint: str, model: str | None, timeout_s: float) -> dict[str, Any]:
    body, boundary = _multipart(item, model)
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_s: float | None = None
    chunks: list[str] = []
    completion_tokens = 0
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            payload = raw_line[6:].strip()
            if payload == b"[DONE]":
                continue
            event = json.loads(payload)
            if event.get("usage"):
                completion_tokens = event["usage"].get("completion_tokens", completion_tokens)
            choices = event.get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content") or ""
                if content and first_token_s is None:
                    first_token_s = time.perf_counter() - started
                chunks.append(content)
    elapsed_s = time.perf_counter() - started
    hypothesis = "".join(chunks)
    if "<asr_text>" in hypothesis:
        hypothesis = hypothesis.rsplit("<asr_text>", 1)[1]
    metric, errors, units = _error_rate(item.reference, hypothesis, item.language)
    return {
        "id": item.id,
        "audio_path": str(item.audio_path),
        "language": item.language,
        "reference": item.reference,
        "hypothesis": hypothesis,
        "audio_duration_s": _duration(item.audio_path),
        "ttft_ms": round(first_token_s * 1000, 3) if first_token_s is not None else None,
        "e2e_ms": round(elapsed_s * 1000, 3),
        "completion_tokens": completion_tokens,
        "metric": metric,
        "errors": errors,
        "reference_units": units,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL: id,audio_path,reference,language")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/audio/transcriptions")
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = [Item(**json.loads(line)) for line in args.manifest.read_text().splitlines() if line.strip()]
    power = PowerSampler()
    power.start()
    wall_started = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(_request, item, args.endpoint, args.model, args.timeout_s) for item in items]
            results = [future.result() for future in futures]
        wall_s = time.perf_counter() - wall_started
    finally:
        power_result = power.stop()

    e2e = [item["e2e_ms"] for item in results]
    ttft = [item["ttft_ms"] for item in results if item["ttft_ms"] is not None]
    errors = sum(item["errors"] for item in results)
    units = sum(item["reference_units"] for item in results)
    audio_s = sum(item["audio_duration_s"] for item in results)
    tokens = sum(item["completion_tokens"] for item in results)
    joules = power_result["joules"] or 0
    report = {
        "schema_version": 1,
        "endpoint": args.endpoint,
        "model": args.model,
        "concurrency": args.concurrency,
        "summary": {
            "requests": len(results),
            "wall_s": round(wall_s, 4),
            "audio_s": round(audio_s, 4),
            "aggregate_realtime_factor": round(wall_s / audio_s, 6),
            "e2e_p50_ms": round(median(e2e), 3),
            "e2e_p95_ms": round(_percentile(e2e, 0.95), 3),
            "ttft_p50_ms": round(median(ttft), 3) if ttft else None,
            "ttft_p95_ms": round(_percentile(ttft, 0.95), 3) if ttft else None,
            "aggregate_error_rate": round(errors / max(1, units), 6),
            "audio_seconds_per_joule": round(audio_s / joules, 6) if joules else None,
            "output_tokens_per_joule": round(tokens / joules, 6) if joules else None,
        },
        "power": power_result,
        "items": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
