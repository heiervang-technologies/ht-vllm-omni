#!/usr/bin/env python3
"""Measure code-only savings available to a dedicated ASR wheel."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    buckets = {
        "diffusion": lambda name: name.startswith("vllm_omni/diffusion/"),
        "non_qwen3_model_plugins": lambda name: name.startswith("vllm_omni/model_executor/models/")
        and "/qwen3_omni/" not in name,
        "qwen3_tts": lambda name: name.startswith("vllm_omni/model_executor/models/qwen3_tts/"),
        "qwen3_omni": lambda name: name.startswith("vllm_omni/model_executor/models/qwen3_omni/"),
    }
    with zipfile.ZipFile(args.wheel) as wheel:
        infos = wheel.infolist()
        measured = {}
        for bucket, predicate in buckets.items():
            selected = [item for item in infos if predicate(item.filename)]
            measured[bucket] = {
                "files": len(selected),
                "uncompressed_bytes": sum(item.file_size for item in selected),
                "compressed_bytes": sum(item.compress_size for item in selected),
            }
        conservative_cut = [
            item
            for item in infos
            if buckets["diffusion"](item.filename) or buckets["non_qwen3_model_plugins"](item.filename)
        ]
        report = {
            "schema_version": 1,
            "wheel": {
                "name": args.wheel.name,
                "bytes": args.wheel.stat().st_size,
                "files": len(infos),
                "uncompressed_bytes": sum(item.file_size for item in infos),
            },
            "buckets": measured,
            "conservative_asr_cut": {
                "compressed_bytes": sum(item.compress_size for item in conservative_cut),
                "percent_of_wheel": round(
                    100 * sum(item.compress_size for item in conservative_cut) / args.wheel.stat().st_size, 2
                ),
                "note": (
                    "Code only; dependency and model-weight savings must be measured by ruby's image-slimming work."
                ),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
