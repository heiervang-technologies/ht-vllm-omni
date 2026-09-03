# Adversarial review: `gems/ruby/qwen3-pascal-runtime`

Reviewed worker tip `4b029d82579759cd324c767793255fc09b5c01ab` against current
`origin/ht` `db0e4590b974f3fd0daecfa785614d3014336e02`. The branch is seven
commits ahead and zero behind (`24 files`, `+976/-81`).

## CLAIMS

### Lean 12 Hz dependency and disk-size profile — UNPROVEN, with a blind gate

The profile wheel itself is real. I independently built it at low priority and
confirmed that its seven non-extra requirements are OmegaConf, SoundFile,
Einops, PrettyTable, aenum, pyzmq, and Janus. The wheel contains the Qwen deploy
YAML and NPY assets. The multi-stage Dockerfile also keeps the checkout, `.git`,
tests, docs, and build-only git out of the final stage. With no `.dockerignore`,
the reported 56,134,435-byte checkout/history-to-wheel logical reduction is
arithmetically reproducible. It is not a measured image-layer saving.
It is also not a measured filesystem-space reduction: no before/after
`docker system df`, image inspect, or host `df` result is committed. The wheel,
checkout, `.git`, and ONNX payload figures are logical-file measurements and
must not be relabeled as freed disk bytes.

The dependency-closure claim is not proved by the submitted gates. This branch
deliberately moves `api_server` behind command dispatch, so
`serve --help` no longer imports the actual HTTP/speech engine graph. Docker CI
still uses only `serve --help` as its import gate
(`.github/workflows/docker-publish.yml:62-73`). The profile exclusion test only
reads requirement strings and AST imports; the workflow does not build the
`qwen3_tts_12hz` profile at all. Consequently, a missing dependency in
`vllm_omni.entrypoints.openai.api_server`, `AsyncOmniEngine`, the speech
handler, or a model-loading child can now pass every submitted packaging test
and the Docker import gate, then fail when `OmniServeCommand.cmd` reaches the
deferred import (`vllm_omni/entrypoints/cli/serve.py:103-110`). This is exactly
the joint failure that dependency slimming must rule out.

Required fix: build the lean profile in a gate, import the deferred API/server
graph inside that image, and perform a bounded Qwen server-start smoke. Keep
the full Pascal model/20-prompt test as a bench-gem acceptance gate.

### 0.6B benchmark handoff — REFUTED for `voice_design`

The new registry entry declares
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` supports `voice_design`, then sends
`task_type: VoiceDesign` (`benchmarks/tts/model_configs.yaml:13-24`). The test
at `tests/packaging/test_qwen3_tts_install_profile.py:43-50` freezes that wrong
mapping instead of validating checkpoint capability.

The checkpoint's published config identifies it as `tts_model_type:
"custom_voice"`, while Qwen publishes VoiceDesign as the separate
`Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint. Qwen's released-model table also
does not mark the 0.6B CustomVoice checkpoint for instruction control. Passing
`task_type: VoiceDesign` skips the CustomVoice speaker path in this repository
and can produce plausible but semantically invalid benchmark audio rather than
a loud capability error. Remove `voice_design` from the 0.6B CustomVoice entry,
or register the actual VoiceDesign checkpoint and validate model type at
request/benchmark construction.

Primary references:

- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/8f9ebcf8826db6eeb9cdd4caa09d575a7f9ce4bd/config.json
- https://github.com/QwenLM/Qwen3-TTS#released-models-description-and-download

### Pascal dtype cleanup — CONFIRMED structurally; runtime result UNPROVEN

The changed talker file no longer forces BF16 and consistently derives prompt,
hidden, speaker, tokenizer, and MTP tensors from the loaded model dtype. The
handoff command does explicitly override both stages to `float32`; therefore
the default deploy YAML's omitted dtype is not a defect in that command.

No Pascal model load, finite-audio check, VRAM result, latency, power, WER, SIM,
or UTMOS data was produced. The branch correctly labels those results pending.
The AST test proves only that the literal `torch.bfloat16` disappeared, not
that a two-stage FP32 checkpoint fits or executes correctly.

### CUDA/cuDNN preflight — PARTLY CONFIRMED, PARTLY REFUTED

The preflight runs before model allocation and loudly rejects a hidden required
GPU, the wrong visible capability, a Torch architecture list without `sm_61`,
CUDA major 13+, and cuDNN above 9.11.1. Seven dependency-free tests passed.

It is not a complete Pascal artifact check:

- `torch.cuda.get_arch_list()` proves only Torch's compiled architectures, not
  that vLLM/custom extension kernels in the image contain and execute `sm_61`.
- The binding fleet policy is CUDA 12.x, but the condition only rejects major
  versions `>=13`; CUDA 11.x would pass. Add an explicit accepted-version test
  if the guard claims to enforce the permanent toolchain policy.
- `_decode_cudnn_version(8902)` reports `(0, 89, 2)`, not cuDNN 8.9.2. This does
  not break the upper-bound decision, but makes the diagnostic report false.

The mandatory bench server start remains necessary and must exercise at least
one real kernel; printing the Torch architecture list is insufficient.

### Codec-buffer churn — CONFIRMED structurally; speed claim UNPROVEN

New frames remain compact CPU `torch.long` tensors, and window construction is
now stack/transpose instead of per-frame `.tolist()` plus nested Python integer
reconstruction. The new mixed list/tensor and storage tests are relevant. No
functional body was deleted in the processor diff. The full Torch test file
could not run on this serving host, and no latency/RSS measurement exists, so
this is a defensible structural improvement rather than a measured speedup.

### Import cuts — CONFIRMED narrowly

The 12 Hz tokenizer selects its modeling module after loading the lightweight
config, so the 25 Hz ONNX/Whisper/VQ implementation is no longer imported by
that wrapper on the 12 Hz path. `diffusers` is type-check-only in
`diffusion/data.py`, and serving no longer imports benchmark/PyDub modules via
the CLI package. These narrow facts are covered by AST tests. Cold-start time
and RSS remain explicitly unmeasured.

### Performance, quality, power, and exact image size — UNPROVEN

There are no committed before/after results on the P5200. The 5% performance
and power bands and WER/SIM/UTMOS deltas are acceptance criteria, not results.
The only raw artifact measurements are host checkout/wheel/ONNX logical sizes;
the ~19.5 GB baseline is forwarded fleet input and the private manifest could
not be read on ruby.

## LOST-BODIES

- Confirmed present after the branch: normal and headless serve dispatch,
  benchmark registration, default full dependency selection, opt-in 12 Hz
  selection, 12 Hz and 25 Hz tokenizer model selection, Base/CustomVoice/
  VoiceDesign prompt bodies, reference-code context, async chunk flush and
  cleanup ownership, deploy YAML and NPY/NPZ wheel assets.
- Confirmed current-base composition: branch is zero commits behind current
  `ht`; no stale-base conflict.
- No production function body was lost in the seven-commit diff.
- Capability correctness was added incorrectly for the new 0.6B benchmark:
  `voice_design` is present as a symbol/configured path but does not belong to
  that checkpoint. This is worse than an absent entry because it can create a
  misleading benchmark result.

## PASCAL

**Compatible in intent, not yet demonstrated.** The branch removes forced BF16,
documents FP32, omits FP8/FA3 from the opt-in dependency metadata, and adds a
useful `sm_61`/CUDA/cuDNN preflight. It does not prove that the complete image's
vLLM kernels execute on `sm_61`, that the attention backend avoids FA2, or that
both FP32 stages fit in 16 GiB. Those are mandatory bench-gem gates, not
assumptions.

## NEW-FAILURE-MODES

- Missing lazy serving dependencies escape CI because the import smoke now
  stops before the graph whose dependencies were slimmed.
- The 0.6B CustomVoice benchmark can silently benchmark the wrong task body.
- A Torch wheel can pass the arch-list check while another compiled extension
  lacks `sm_61`; failure moves from startup preflight to model load/first op.
- CUDA 11.x is accepted despite the stated CUDA-12.x policy.
- Malformed or unexpected tokenizer config now fails earlier with an explicit
  unsupported-model-type error. This is a loud and acceptable new failure.
- Tensor frame/ref-code shape mismatches now fail loudly rather than being
  flattened through Python lists. This is preferable.

## VERDICT

**mergeable-with-fixes**. The structural code is directionally sound and the
claims are mostly disciplined, but correct the 0.6B task registry and restore a
dependency gate that actually imports/starts the deferred serving path in the
lean image. Do not publish image-size, speed, power, quality, or Pascal-runtime
claims until the documented P5200 gates produce committed raw results.

**UPSTREAM: needs-split.** The model-dtype cleanup, tensor-native codec window,
tokenizer/CLI lazy imports, package-data fix, and generic runtime-size audit are
plausible upstream candidates once their runtime gates pass. The Pascal
`sm_61`/CUDA/cuDNN policy, fleet-specific artifact narrative, HT Docker
workflow, and local benchmark rollout gates belong in the public HT fork.
Separate those concerns before dispatcher/Markus considers any outward
submission; this review does not authorize one.
