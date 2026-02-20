#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate audio samples to verify generate_codes() produces correct output.

Loads the real Qwen3-TTS model and generates speech through the full pipeline:
  talker → code predictor (generate_codes()) → speech tokenizer decoder

Usage:
    python benchmarks/audio_quality_test.py [--model MODEL] [--device DEVICE]

Output WAV files are saved to output_audio/ for manual listening.
"""

import argparse
import logging
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Bootstrap: stub vllm modules so qwen3_tts.py can be imported standalone.
# The actual TTS model uses only transformers/torch — vllm is only needed
# by the Qwen3TTSModelForGeneration wrapper which we don't use here.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]

_STUB_FQNS = [
    "vllm", "vllm.config", "vllm.logger", "vllm.sequence",
    "vllm_omni", "vllm_omni.patch",
    "vllm_omni.diffusion", "vllm_omni.diffusion.compile",
    "vllm_omni.model_executor",
    "vllm_omni.model_executor.model_loader",
    "vllm_omni.model_executor.model_loader.weight_utils",
    "vllm_omni.model_executor.models",
    "vllm_omni.model_executor.models.output_templates",
    "vllm_omni.model_executor.models.qwen3_omni",
    "vllm_omni.model_executor.models.registry",
    "vllm_omni.model_executor.models.qwen3_tts",
]


def _setup_stubs():
    from typing import NamedTuple

    for fqn in _STUB_FQNS:
        if fqn not in sys.modules:
            mod = types.ModuleType(fqn)
            mod.__path__ = [str(_REPO / fqn.replace(".", "/"))]
            mod.__package__ = fqn
            mod.__spec__ = None
            sys.modules[fqn] = mod

    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)
    sys.modules["vllm.config"].VllmConfig = type("VllmConfig", (), {})

    class IntermediateTensors:
        def __init__(self, d=None):
            self.tensors = d or {}
    sys.modules["vllm.sequence"].IntermediateTensors = IntermediateTensors

    class OmniOutput(NamedTuple):
        text_hidden_states: object
        multimodal_outputs: dict | None = None
        intermediate_tensors: object | None = None
        next_token_id: object | None = None
    sys.modules["vllm_omni.model_executor.models.output_templates"].OmniOutput = OmniOutput

    sys.modules["vllm_omni.diffusion.compile"].regionally_compile = lambda *a, **kw: None
    sys.modules["vllm_omni.model_executor.model_loader.weight_utils"].download_weights_from_hf_specific = (
        lambda *a, **kw: None
    )
    sys.modules["vllm_omni.model_executor.models.qwen3_omni"].Qwen3OmniMoeForConditionalGeneration = type(
        "Stub", (), {}
    )
    sys.modules["vllm_omni.model_executor.models.registry"].OmniModelRegistry = type("Stub", (), {})

    for fqn in _STUB_FQNS:
        parts = fqn.split(".")
        if len(parts) > 1:
            parent = sys.modules.get(".".join(parts[:-1]))
            child = sys.modules.get(fqn)
            if parent and child:
                setattr(parent, parts[-1], child)


_setup_stubs()


def main():
    parser = argparse.ArgumentParser(description="TTS audio quality test")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="output_audio")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    t0 = time.perf_counter()

    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
    )
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print()

    # ── Test samples ────────────────────────────────────────────────────
    samples = [
        {
            "name": "english_hello",
            "text": "Hello! This is a test of the text to speech system. The quick brown fox jumps over the lazy dog.",
            "speaker": "Vivian",
            "language": "English",
        },
        {
            "name": "english_numbers",
            "text": "One, two, three, four, five. The year is twenty twenty six.",
            "speaker": "Vivian",
            "language": "English",
        },
    ]

    # ── Generate ────────────────────────────────────────────────────────
    for sample in samples:
        name = sample["name"]
        print(f"Generating: {name}")
        print(f"  Text: {sample['text'][:60]}{'...' if len(sample['text']) > 60 else ''}")

        t0 = time.perf_counter()
        with torch.no_grad():
            wavs, sr = model.generate_custom_voice(
                text=sample["text"],
                speaker=sample["speaker"],
                language=sample["language"],
            )
        gen_time = time.perf_counter() - t0

        wav = wavs[0]
        duration = len(wav) / sr
        rms = np.sqrt(np.mean(wav.astype(np.float64) ** 2))

        out_path = out_dir / f"{name}.wav"
        sf.write(str(out_path), wav, sr)

        print(f"  Duration:    {duration:.2f}s")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Samples:     {len(wav):,}")
        print(f"  RMS:         {rms:.4f}")
        print(f"  Min/Max:     {wav.min():.4f} / {wav.max():.4f}")
        print(f"  Gen time:    {gen_time:.2f}s")
        print(f"  RTF:         {gen_time / duration:.2f}x")
        print(f"  Saved:       {out_path}")
        print()

    print("Done. Listen to the WAV files to verify audio quality.")


if __name__ == "__main__":
    main()
