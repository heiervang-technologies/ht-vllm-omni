# Qwen3-ASR optimization plan for Pascal gems

## Status and ownership

Measured draft, 2026-08-15. The CPU measurements and recommendations in
this document are from **amber** and are attributed to the amber ASR mission.
Ruby owns dependency/container slimming and TTS; the ASR-specific artifact cut
list below is an input to ruby's work, not a competing dependency change.

The code path was mapped against this fork at `db0e4590` and vLLM 0.20.0 at
`88d34c64`. The model and evaluation revisions are pinned in
`benchmarks/asr/qwen3_asr_pascal_bench.yaml`.

## Request path

```text
multipart upload bytes
  -> compressed-size guard                         CPU / event-loop
  -> libsndfile decode to native-rate float32      CPU
  -> stereo-to-mono reduction                      CPU
  -> libswresample to 16 kHz                       CPU
  -> optional 30 s low-energy split                CPU
  -> pad to 160-sample Whisper hop                 CPU
  -> 128-bin Whisper log-mel                       CPU
  -> audio tower: 3 convs + 24 transformer layers GPU
  -> Qwen3 28-layer autoregressive decode          GPU
  -> token decode, ASR-tag removal, HTTP/SSE       CPU
```

The standard upload endpoint materializes the entire decoded/resampled
waveform before it decides whether to split it. The realtime endpoint instead
buffers 5 seconds before each inference request. Its current upstream buffer
preallocates 60 seconds (3.84 MB at 16 kHz float32) and shifts the remaining
samples after every emitted segment.

## Reproducible CPU latency budget

Run from the repository root:

```bash
nice -n 10 ~/work/.venv-asr/bin/python \
  benchmarks/asr/bench_qwen3_asr_preprocessing.py \
  --work-dir ~/work/asr-bench-corpus \
  --output benchmarks/asr/results/amber-rerun.json \
  --repeats 9
```

The script deterministically generates the fixed corpus and records SHA-256
digests. Raw before/after numbers are committed under `benchmarks/asr/results`.

| Input | Stage | Baseline p50 | Fast path p50 | Result |
|---|---|---:|---:|---:|
| 5 s, 16 kHz mono | complete CPU path | 11.40 ms | 11.41 ms | neutral |
| 30 s, 48 kHz stereo | mono reduction | 22.78 ms | 1.46 ms | -93.6% |
| 30 s, 48 kHz stereo | complete CPU path | 80.59 ms | 59.19 ms | **-26.6%** |
| 120 s, 44.1 kHz stereo | mono reduction | 79.94 ms | 5.45 ms | -93.2% |
| 120 s, 44.1 kHz stereo | complete CPU path | 317.19 ms | 241.48 ms | **-23.9%** |

The optimized two-channel add/scale is bit-identical to NumPy's float32 mean
on the fixed corpus. Feature extraction remains the largest CPU stage at about
1.45 ms per second of audio on long inputs. Hop padding is effectively free;
caching or complicating it cannot change a deployment decision.

The fused PyAV decode/resample alternative was rejected: it regressed 5-second
and 30-second inputs and only slightly helped the 120-second native-rate
ingest. The ring-buffer alternative was also rejected. It saved 3.52 MB idle
and improved a hostile 10-minute single write by 67%, but slowed normal 20 ms
stream writes by 19.8% (about 20 ms total over ten minutes). Bounded client
chunks are the simpler failure mode.

## GPU latency estimate and required measurement

Amber cannot run the model because its live llama worker owns the GPU. These
are planning estimates, not benchmark claims:

- A 30-second clip produces about 3,000 mel frames and about 390 post-CNN audio
  tokens.
- The 24-layer, width-1024 encoder has roughly 236 GFLOP of linear work for
  those 390 tokens, excluding convolution and attention-score work.
- On the P5200, a deliberately broad planning range is 0.12-0.50 seconds for
  the fp32 encoder and roughly 2-7 ms per autoregressive output token. Actual
  kernel efficiency and attention backend decide the result.
- A static INT8 encoder that really reaches Pascal `dp4a` should target
  0.02-0.10 seconds for the same encoder workload. A weight-only path that
  dequantizes into fp16 does not qualify.

The fixed bench procedure below must replace every range with encoder, decode,
TTFT, end-to-end, VRAM/RSS, and power measurements.

## Chunking and concurrency

The first partial result cannot arrive before a realtime chunk closes. Test
2.5, 5, 10, and 30 seconds:

| Chunk | Expected effect | Accuracy risk |
|---:|---|---|
| 2.5 s | Lowest acquisition/partial latency | Highest boundary and context loss |
| 5 s | Current realtime default | Moderate boundary loss |
| 10 s | Better sentence context | Higher TTFT and per-request encoder burst |
| 30 s | Best full-context baseline | Not interactive; larger transient tensors |

Do not buffer a long upload merely to split it afterward. The next structural
change should decode and resample incrementally into bounded 30-second chunks,
with 0.5-1.0 seconds of left context evaluated for WER and duplicate-text
handling. The client-facing stream should cap a single buffered append at 60
seconds and reject over-limit input loudly.

The bench matrix uses 5/30/120-second audio at 50/35/15%, concurrency
1/2/4/8, and `max_num_seqs` 1/2/4/8. Stop a run before the shared 16 GB host
ceiling becomes pressure: process RSS 12 GiB, GPU memory 14.5 GiB, or any
request error. A 30-second clip is about 390 encoder tokens, so concurrency 4
starts around 1,560 audio tokens before text/decode tokens. This is a starting
shape, not a recommended production setting until measured.

## Pascal INT8 plan

Qwen3-ASR's checkpoint declares bfloat16. Pascal cannot execute bfloat16, and
casting this checkpoint to fp16 is not an honest fallback; the baseline is
fp32. The startup guard now rejects CUDA newer than cu126, wheels without
`sm_61`, bf16 execution, and fp16 casts of a bf16-declared checkpoint.

Quantize these audio-tower linears:

- `conv_out`;
- every layer's `self_attn.qkv` and `self_attn.out_proj`;
- every layer's `fc1` and `fc2`;
- `proj1` and `proj2`.

Keep the three convolutions, LayerNorms, positional addition, and attention
softmax out of the first INT8 experiment. The repository's existing
`DiffusionInt8Config` is explicitly not a candidate: it reports minimum
capability 8.0. The primary Pascal candidate is ONNX Runtime static QDQ INT8
with calibration. TorchAO weight-only is probe-only and must be rejected unless
the profiler proves an INT8 `dp4a` kernel and a latency win.

Accuracy gates are overall WER regression <=0.5 absolute points and each
language <=1.0 point, using pinned LibriSpeech dummy and FLEURS English,
Norwegian, German, and Mandarin splits. Mandarin uses CER. The calibration and
evaluation slices do not overlap.

## Input hardening

The baseline accepted a WAV claiming roughly 2 GiB of data and an actually
truncated WAV. The fork loader now:

- rejects incomplete classic RIFF containers before codec entry;
- rejects zero-sample audio;
- caps declared decoded float32 PCM at 512 MiB by default through
  `VLLM_OMNI_MAX_DECODED_AUDIO_MB`;
- preserves vLLM's PyAV fallback for non-libsndfile containers;
- maps failures through the existing request validation boundary.

The committed fuzz cases all reject after the change. The new failure mode is
intentional: a valid file whose decoded PCM exceeds 512 MiB returns an input
error and must be sent as bounded chunks instead. PyAV fallback formats are
checked after decode in this patch; a follow-up should enforce the byte budget
during frame iteration before exposing arbitrary compressed formats to paid
traffic.

## Artifact cut list

The measured full wheel is 2,626,388 bytes (9,515,300 bytes uncompressed).
A conservative ASR cut removes diffusion and non-Qwen3 model plugins:
1,754,285 compressed bytes, or **66.79% of the wheel**. Diffusion alone is
947,525 compressed bytes; non-Qwen3 model plugins are 806,760. Qwen3-TTS code
is 77,184 compressed bytes and belongs on ruby's TTS side, not the ASR image.

This is code-only. Dependency, container-layer, and model-weight savings are
deliberately left to ruby's slimming work. A dedicated Qwen3-ASR image should
load vLLM's native Qwen3-ASR implementation and must not download or bake
talker, code2wav, diffusion, or TTS tokenizer weights.

Reproduce the code measurement with:

```bash
uv build --wheel --python ~/work/.venv-asr/bin/python --out-dir ~/work/asr-artifacts
python benchmarks/asr/measure_qwen3_asr_artifact.py \
  ~/work/asr-artifacts/vllm_omni-*.whl \
  --output benchmarks/asr/results/amber-artifact-rerun.json
```

## Ready for bench

Prepare the pinned, non-overlapping datasets, then start the fp32 server using
a cu126/sm_61 build:

```bash
python benchmarks/asr/prepare_qwen3_asr_eval.py \
  --output-dir ~/work/qwen3-asr-eval
```

For each quantization candidate and each row in
`qwen3_asr_pascal_bench.yaml`, run:

```bash
nice -n 10 python benchmarks/asr/bench_qwen3_asr_serving.py \
  --manifest ~/work/qwen3-asr-eval/evaluation.jsonl \
  --endpoint http://127.0.0.1:8000/v1/audio/transcriptions \
  --model Qwen/Qwen3-ASR-1.7B \
  --concurrency 1 \
  --output qwen3-asr-fp32-c1.json
```

Repeat at concurrency 2/4/8 and for INT8. The script records streaming TTFT,
end-to-end latency, aggregate real-time factor, WER/CER counts, output tokens,
100 ms power samples, audio-seconds/joule, tokens/joule, and peak VRAM. Commit
the raw JSON. Cancel INT8 if it misses either accuracy gate, does not execute
`dp4a`, or improves encoder latency by less than 20%.
