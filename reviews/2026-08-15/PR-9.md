# Adversarial review — PR #9 diffusion FP8

Target: https://github.com/heiervang-technologies/ht-vllm-omni/pull/9  
Reviewed head: `9d8ccedf91fe7e047039ed9c764a62a4a050ab93`  
Live base at review: `ht` at `be72432dda44c8b0cf0dbf6f095885723f24e9b5`

## CLAIMS

- **“Adds FP8 W8A8 diffusion infrastructure” — REFUTED as a current merge claim.** The old head contains an FP8 wrapper and threads it into several DiTs, but the live `ht` base already has a newer generalized `vllm_omni.quantization` framework with FP8 plus component routing, ModelOpt handling, INT8, GGUF, INC/AutoRound, docs, and broader tests. This PR is obsolete, not additive.
- **“CPU-init/offload makes quantized models fit” — UNPROVEN.** The description's test boxes are all unchecked, no CI checks are attached, and no committed before/after memory data exists. The loader also treats any non-null `od_config.quantization` as a reason to force both CPU and layerwise offload without a per-model compatibility gate.
- **“FP8 path is safely selectable” — REFUTED for Pascal.** `get_min_capability()` exists but has no runtime caller. Config resolution accepts `quantization="fp8"` without checking the device, and loader post-processing moves modules to the target GPU and invokes FP8 conversion. The only explicit capability check is inside the smoke test, where GPUs below 8.9 are skipped. A test skip is not a deployment gate.
- **“Tests validate the feature” — UNPROVEN.** Reviewer static check parsed 51 changed Python files with zero AST failures. The PR has no recorded test results or raw performance/quality data; its GPU smoke explicitly avoids unsupported hardware.
- **“This is the current FP8 PR” — REFUTED.** PR #21 points to the exact same head SHA. Two open PRs do not represent complementary implementations.

## LOST-BODIES

- Present in the old head: FP8 wrapper/factory, config fields, DiT wiring across named models, a metadata wiring test, a high-end-GPU smoke path, example flags, CPU-init/offload logic.
- Missing compared with current `ht`: the generalized `vllm_omni.quantization` factory, per-component routing, ModelOpt checkpoint handling, Pascal-relevant INT8, GGUF, INC/AutoRound, current component/int8/gguf/quality tests, and current quantization documentation.
- Staleness: 447 base commits behind, 13 head-only commits (the first seven are unrelated Qwen3-TTS/branding work), and 41 synthetic-merge conflicts.

## PASCAL

**incompatible-because-fp8-does-not-exist-on-sm_61-and-selection-is-not-gated.** Fleet deployment must reject FP8 before model construction and preserve INT8 dp4a as the acceleration path.

## NEW-FAILURE-MODES

- Pascal accepts the flag, then fails late during FP8 allocation/kernel selection.
- Blanket offload mutation changes runtime behavior for every configured quantization method.
- Conflict resolution toward the PR head can delete the live INT8/GGUF/component-routing framework.
- Duplicate PRs can receive inconsistent review/merge decisions for identical code.

## VERDICT

**do-not-merge.** Close as obsolete. If any missing FP8 behavior remains after auditing current `ht`, submit one clean PR against current `ht` with an early hardware gate and an explicit `sm_61 -> fail loud / never select FP8` test.
