# Qwen3-TTS optimization plan for Pascal

## Status

Active implementation plan, refreshed from `docs/optimization-plan-rfc` on
2026-08-15. The old RFC described a monolithic Qwen3-TTS implementation. The
current `ht` branch already has the two-stage talker → Code2Wav pipeline,
continuous batching, async codec chunks, and progressive PCM output. Those are
the baseline, not future work.

The target is Quadro P5200 (`sm_61`) with 16 GiB VRAM and 16 GB host RAM. It is
a Pascal deployment: CUDA 12.x only, a torch build containing `sm_61`, cuDNN no
newer than 9.11.1, no BF16/FP8, and no FlashAttention-2/SageAttention.

## Current serving path

1. `OmniOpenAIServingSpeech` validates the request, prepares text/reference
   audio, and starts the engine generator.
2. `Qwen3TTSTalkerForConditionalGeneration` builds prompt embeddings and runs
   the AR talker plus the serial residual-codebook predictor.
3. `talker2code2wav_async_chunk` accumulates one RVQ frame at a time. The first
   window can be one frame for time-to-first-audio; later windows use 25 frames
   plus left/reference context.
4. The shared-memory connector sends flattened codebook-major integer tokens to
   `Qwen3TTSCode2Wav`.
5. Code2Wav decodes FP32 waveform chunks. The speech service converts them to
   signed 16-bit PCM and streams the bytes.

The remaining latency structure is therefore AR/code-predictor time to the
first codec frame, connector polling (currently 10 ms), first-frame Code2Wav,
and HTTP chunk delivery. The processor previously created Python integers with
`cpu().tolist()` for every frame, rebuilt a nested list window, then allocated
another `torch.long` tensor for every emitted chunk. It now retains compact CPU
tensors and stacks/transposes the selected window, including reference frames,
without the nested Python-int round trip. The unavoidable device-to-host copy
remains one explicit compact copy per frame. A bounded ring buffer is a later
candidate; it also needs coordinated cleanup and absolute-frame counters.

## Measured artifact budget

Raw commands and outputs are committed in
`benchmarks/results/ruby-runtime-audit-2026-08-15.md`.

| Item | Current/baseline | Proposed | Logical saving | Confidence |
|---|---:|---:|---:|---|
| Complete Pascal runtime | ~19.5 GB | pending bench-image build | pending | Baseline supplied by fleet mission; private GHCR manifest returned 401 from ruby |
| Clean tracked checkout copied by old image | 28,203,657 B | 9,512,094 B installed wheel | 18,691,563 B | Measured locally |
| `.git` copied by old image | 37,442,872 B | 0 B | 37,442,872 B | Measured locally |
| Source + history → wheel | 65,646,529 B | 9,512,094 B | **56,134,435 B** | Measured logical bytes; layer compression differs |
| 25 Hz `onnxruntime==1.23.2` tail | 17,382,649 B wheel / 49,610,946 B files | 0 B in a dedicated 12 Hz recipe | up to 49,610,946 B | Measured wheel; generic image must retain 25 Hz support |

`Dockerfile.slim` now builds a wheel in a builder stage. The final stage contains
neither repository history/source/tests/docs nor build-only git, and installs
`.` rather than `.[dev]`. Removing the dev extra also removes a CUDA-13-only
Mooncake dependency, test/quality stacks, and their transitives from the serving
recipe. The exact dev-extra saving must be read from the built image because the
vLLM base already supplies an unknown subset; do not sum wheel sizes and call it
an image saving.

### Serving dependency classification

| Class | Packages/components | Decision |
|---|---|---|
| Core | vLLM, torch, torchaudio, transformers, numpy, soundfile, OmegaConf, pyzmq, janus, einops | Keep; imported by engine, audio, pipeline, or connector paths |
| 12 Hz Qwen3-TTS | Mimi/transformers decoder and SoX format libraries | Keep |
| 25 Hz only | onnxruntime and Whisper/VQ implementation | Lazy-imported; omit only from a dedicated 12 Hz image |
| Diffusion-only | diffusers, cache-dit, torchsde, image/video helpers | Candidate omission in a Qwen3-TTS-only recipe; generic Omni image keeps them |
| Dev/quality | pytest, datasets, mypy, pre-commit, Whisper evaluation, OpenCV, Mooncake CUDA 13, WER/SIM/UTMOS tools | Never install in runtime |
| Triton / CUDA Python libraries | supplied transitively by torch/vLLM base | Do not delete blindly: Triton cannot generate useful Pascal kernels, but vLLM imports may still require the package |
| FA3/FlashAttention/SageAttention | unsupported on `sm_61` | Omit from Pascal recipe; use PyTorch SDPA math/memory-efficient paths |

Before pruning CUDA libraries, inventory both `site-packages/nvidia/**/lib` and
`/usr/local/cuda*/lib*`, resolve every loaded object with `ldd`, and compare
inodes/hashes. A same-named library is not proof that either copy is unused.

## Import and cold-start budget

Ruby cannot reproduce the serving import baseline: the host has Python 3.14
while the project requires `<3.14`, and no serving environment is installed.
The measured attempt exited in 0.048 s at the first missing dependency
(`aenum`). This is recorded as **could not measure**, not as a 48 ms cold start.

One verified import-graph cut is implemented: importing the Qwen3-TTS tokenizer
wrapper no longer imports both the 12 Hz and 25 Hz model graphs. It reads the
lightweight config first and imports only the selected implementation. This
keeps `onnxruntime` and the 25 Hz Whisper/VQ graph out of a 12 Hz process import.
The bench image must capture `python -X importtime` before and after to quantify
wall time and RSS.

The entrypoint itself still imports `api_server` at module scope through
`cli.serve`; that pulls FastAPI, serving implementations, diffusion detection,
and vLLM server code before argument dispatch. Moving `omni_run_server` into
`OmniServeCommand.cmd` is a later cold-start candidate, gated by CLI/help and
forkserver tests.

## Pascal precision policy

| Stage/data | Pascal representation | Reason |
|---|---|---|
| Talker and code predictor weights/compute | FP32 | BF16 is unsupported; FP16 compute is 1/64 rate and only useful as storage |
| Prompt, hidden, speaker, and tokenizer tensors | Engine/model dtype; FP32 when Pascal stage is configured FP32 | Prevents prior hard-coded BF16 casts from corrupting an otherwise FP32 run |
| Code2Wav decoder and waveform | FP32 | Already explicitly configured by Code2Wav; retain for codec quality |
| RVQ codes | integer; `torch.long` only at embedding/API boundary | No benefit from floating storage |
| HTTP PCM | signed 16-bit bytes | Wire format, not model compute |

The talker now follows the actual model dtype for prefill/decode, speaker
extraction, reference tokenizer loading, and the MTP fast path. A Pascal bench
must explicitly configure both stages as FP32 and record peak VRAM. If the
checkpoint does not fit, report that honestly; do not cast a BF16-trained stage
to FP16 as a workaround.

The `feat/diffusion-fp8-quantization` branch is a dead end for this target. FP8
needs newer hardware and is unrelated to the INT8 `dp4a` path available on
Pascal. Qwen3-TTS has no validated GGUF/INT8 execution path in this runtime, so
no INT8 speed claim is made here.

## Startup hardening

Every Omni serve command now runs a CUDA compatibility preflight before model
allocation. On a visible `sm_61` device it checks:

- the GPU really reports capability 6.1;
- `torch.cuda.get_arch_list()` contains `sm_61` (or equivalent PTX entry);
- torch uses CUDA 12.x, not CUDA 13;
- cuDNN is at most 9.11.1.

`VLLM_OMNI_REQUIRED_CUDA_ARCH=sm_61` forces the same validation in a container
gate. The new failure mode is intentional: a missing/hidden CUDA device aborts
when that override is set, rather than allowing an image-only smoke test to
pass without checking its target architecture.

## Bench acceptance

Run the exact commands in `PROGRESS.md` on a non-serving Pascal bench gem.
Accept only if all are true:

1. preflight and CLI import gate pass;
2. 20-prompt CustomVoice benchmark has no audio-quality regression and reports
   audio TTFP, E2E, RTF, peak VRAM, average watts, and tokens/audio-second per
   joule where available;
3. Base voice-clone smoke proves the FP32 speaker/reference path is finite and
   non-silent;
4. image size and `site-packages` audit show measured reductions;
5. restart/rebuild reproduces the checks—no runtime-only mutation.
