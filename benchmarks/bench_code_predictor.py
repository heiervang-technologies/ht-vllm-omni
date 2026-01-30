#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: generate_codes() (manual KV-cache loop) vs HF generate().

Usage:
    python benchmarks/bench_code_predictor.py [--device cpu|cuda] [--iters N]

Measures wall-clock time per call for the code predictor's 31-step
autoregressive decode, comparing the manual loop against HuggingFace's
GenerationMixin.generate().
"""

import argparse
import logging
import sys
import time
import types
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Bootstrap stubs (same as test suite)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]

_STUB_FQNS = [
    "vllm",
    "vllm.config",
    "vllm.logger",
    "vllm.sequence",
    "vllm_omni",
    "vllm_omni.patch",
    "vllm_omni.model_executor",
    "vllm_omni.model_executor.model_loader",
    "vllm_omni.model_executor.model_loader.weight_utils",
    "vllm_omni.model_executor.models",
    "vllm_omni.model_executor.models.qwen3_omni",
    "vllm_omni.model_executor.models.registry",
    "vllm_omni.model_executor.models.qwen3_tts",
]


def _setup_stubs():
    for fqn in _STUB_FQNS:
        if fqn not in sys.modules:
            mod = types.ModuleType(fqn)
            mod.__path__ = [str(_REPO / fqn.replace(".", "/"))]
            mod.__package__ = fqn
            mod.__spec__ = None
            sys.modules[fqn] = mod

    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)
    sys.modules["vllm.config"].VllmConfig = type("VllmConfig", (), {})
    sys.modules["vllm_omni.model_executor.model_loader.weight_utils"].download_weights_from_hf_specific = (
        lambda *a, **kw: None
    )
    sys.modules["vllm_omni.model_executor.models.qwen3_omni"].Qwen3OmniMoeForConditionalGeneration = type(
        "Stub", (), {}
    )
    sys.modules["vllm_omni.model_executor.models.registry"].OmniModelRegistry = type("Stub", (), {})

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope(config=None, device=None, seq_len=None, layer_type=None):
            base = getattr(config, "rope_theta", 10000.0)
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
            )
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _default_rope

    for fqn in _STUB_FQNS:
        parts = fqn.split(".")
        if len(parts) > 1:
            parent = sys.modules.get(".".join(parts[:-1]))
            child = sys.modules.get(fqn)
            if parent and child:
                setattr(parent, parts[-1], child)


_setup_stubs()

from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

# Production 0.6B code predictor dimensions
PROD_DIMS = dict(
    vocab_size=2048,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=5,
    num_attention_heads=16,
    num_key_value_heads=8,
    num_code_groups=32,
    max_position_embeddings=32768,
    use_cache=True,
    pad_token_id=0,
)


def build_model(device: torch.device):
    cp_config = Qwen3TTSTalkerCodePredictorConfig(**PROD_DIMS)
    talker_config = Qwen3TTSTalkerConfig(
        vocab_size=PROD_DIMS["vocab_size"],
        hidden_size=PROD_DIMS["hidden_size"],
        intermediate_size=PROD_DIMS["intermediate_size"],
        num_hidden_layers=PROD_DIMS["num_hidden_layers"],
        num_attention_heads=PROD_DIMS["num_attention_heads"],
        num_key_value_heads=PROD_DIMS["num_key_value_heads"],
        num_code_groups=PROD_DIMS["num_code_groups"],
        text_hidden_size=PROD_DIMS["hidden_size"],
        code_predictor_config=cp_config,
    )
    model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        config=cp_config, talker_config=talker_config,
    )
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _measure(fn, n_iters, device):
    """Run fn() n_iters times and return list of elapsed ms."""
    times = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
    return times


def time_generate_codes(model, embeds, n_iters, device, sampling_kwargs):
    return _measure(
        lambda: model.generate_codes(embeds, **sampling_kwargs),
        n_iters, device,
    )


def time_hf_generate(model, embeds, n_iters, n_codes, device, do_sample):
    hf_kwargs = dict(
        inputs_embeds=embeds,
        max_new_tokens=n_codes,
        do_sample=do_sample,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    if do_sample:
        hf_kwargs.update(top_k=50, top_p=1.0, temperature=0.9)
    return _measure(
        lambda: model.generate(**hf_kwargs),
        n_iters, device,
    )


def stats(times):
    t = torch.tensor(times)
    return t.mean().item(), t.std().item(), t.median().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark code predictor")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()

    device = torch.device(args.device)
    n_codes = PROD_DIMS["num_code_groups"] - 1  # 31

    print(f"Device:       {device}")
    if device.type == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"Code groups:  {PROD_DIMS['num_code_groups']} (generates {n_codes} codes)")
    print(f"Hidden size:  {PROD_DIMS['hidden_size']}")
    print(f"Layers:       {PROD_DIMS['num_hidden_layers']}")
    print(f"Warmup:       {args.warmup}")
    print(f"Iterations:   {args.iters}")
    print()

    print("Building model...", flush=True)
    model = build_model(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {param_count:,}")
    print()

    torch.manual_seed(42)
    embeds = torch.randn(1, 2, PROD_DIMS["hidden_size"], device=device)

    scenarios = [
        ("Greedy (do_sample=False)", dict(do_sample=False, top_p=1.0, top_k=0, temperature=1.0), False),
        ("Sampling (top_k=50, t=0.9)", dict(do_sample=True, top_p=1.0, top_k=50, temperature=0.9), True),
    ]

    for label, gc_kwargs, hf_do_sample in scenarios:
        print(f"--- {label} ---")

        # Warmup
        print(f"  Warming up ({args.warmup} iters)...", flush=True)
        with torch.no_grad():
            time_generate_codes(model, embeds, args.warmup, device, gc_kwargs)
            time_hf_generate(model, embeds, args.warmup, n_codes, device, hf_do_sample)

        # Timed runs
        print(f"  Benchmarking ({args.iters} iters)...", flush=True)
        with torch.no_grad():
            gc_times = time_generate_codes(model, embeds, args.iters, device, gc_kwargs)
            hf_times = time_hf_generate(model, embeds, args.iters, n_codes, device, hf_do_sample)

        gc_mean, gc_std, gc_med = stats(gc_times)
        hf_mean, hf_std, hf_med = stats(hf_times)
        speedup = hf_mean / gc_mean

        print()
        print(f"  {'Method':<25} {'Mean (ms)':>10} {'Std (ms)':>10} {'Median (ms)':>12}")
        print(f"  {'-' * 57}")
        print(f"  {'generate_codes()':<25} {gc_mean:>10.2f} {gc_std:>10.2f} {gc_med:>12.2f}")
        print(f"  {'HF generate()':<25} {hf_mean:>10.2f} {hf_std:>10.2f} {hf_med:>12.2f}")
        print(f"  Speedup: {speedup:.2f}x  ({(1 - 1/speedup) * 100:.1f}% faster)")
        print()


if __name__ == "__main__":
    main()
