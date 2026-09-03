#!/usr/bin/env python3
"""Materialize the pinned Qwen3-ASR WER/CER and INT8 calibration sets."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, load_dataset

LIBRISPEECH_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_LANGUAGES = {
    "en_us": "en",
    "nb_no": "no",
    "de_de": "de",
    "cmn_hans_cn": "zh",
}


def _write_audio(raw_audio: dict[str, Any], output: Path) -> None:
    source = io.BytesIO(raw_audio["bytes"]) if raw_audio.get("bytes") is not None else raw_audio["path"]
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")


def _rows(dataset, limit: int | None = None):
    ids = dataset["id"]
    indices = sorted(range(len(ids)), key=lambda index: str(ids[index]))
    if limit is not None:
        indices = indices[:limit]
    return (dataset[index] for index in indices)


def _materialize(
    dataset,
    *,
    output_dir: Path,
    prefix: str,
    language: str,
    text_key: str,
    limit: int | None,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(_rows(dataset, limit)):
        item_id = f"{prefix}-{row['id']}"
        audio_path = output_dir / f"{index:04d}-{row['id']}.wav"
        _write_audio(row["audio"], audio_path)
        manifest.append(
            {
                "id": item_id,
                "audio_path": str(audio_path.resolve()),
                "reference": row[text_key],
                "language": language,
            }
        )
    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluation: list[dict[str, str]] = []
    calibration: list[dict[str, str]] = []

    librispeech = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
        revision=LIBRISPEECH_REVISION,
    ).cast_column("audio", Audio(decode=False))
    evaluation.extend(
        _materialize(
            librispeech,
            output_dir=args.output_dir / "evaluation/librispeech",
            prefix="librispeech",
            language="en",
            text_key="text",
            limit=None,
        )
    )

    for fleurs_config, language in FLEURS_LANGUAGES.items():
        train = load_dataset(
            "google/fleurs",
            fleurs_config,
            split="train",
            revision=FLEURS_REVISION,
        ).cast_column("audio", Audio(decode=False))
        validation = load_dataset(
            "google/fleurs",
            fleurs_config,
            split="validation",
            revision=FLEURS_REVISION,
        ).cast_column("audio", Audio(decode=False))
        calibration.extend(
            _materialize(
                train,
                output_dir=args.output_dir / f"calibration/{fleurs_config}",
                prefix=f"fleurs-{fleurs_config}-train",
                language=language,
                text_key="transcription",
                limit=32,
            )
        )
        evaluation.extend(
            _materialize(
                validation,
                output_dir=args.output_dir / f"evaluation/{fleurs_config}",
                prefix=f"fleurs-{fleurs_config}-validation",
                language=language,
                text_key="transcription",
                limit=100,
            )
        )

    _write_jsonl(args.output_dir / "calibration.jsonl", calibration)
    _write_jsonl(args.output_dir / "evaluation.jsonl", evaluation)
    print(f"calibration={len(calibration)} evaluation={len(evaluation)} output={args.output_dir}")


if __name__ == "__main__":
    main()
