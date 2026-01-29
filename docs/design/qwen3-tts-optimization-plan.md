# Qwen3 TTS Optimization Plan

## Status

**Draft** — RFC for discussion

## Summary

This document describes a plan to optimize Qwen3 TTS inference in vLLM-Omni. The current implementation runs as a single monolithic stage with `max_batch_size=1`, no streaming support, and a sequential code predictor bottleneck that requires 31 transformer forward passes per generated token. The proposed changes decompose the pipeline into multiple stages, enable batched inference, add streaming audio output, and optimize the dominant code predictor bottleneck.

## Current Architecture

Qwen3 TTS runs as a single `OmniLLM` stage:

```
Text → [Talker (20L transformer)] → first codebook logits
                ↓
        [Code Predictor (5L transformer)] × 31 sequential passes → codebooks 2-32
                ↓
        [Speech Tokenizer Decoder] → waveform
```

### Current Stage Config (`qwen3_tts.yaml`)

- 1 stage, 1 GPU, `max_batch_size: 1`
- `gpu_memory_utilization: 0.1`
- `enforce_eager: true` (no CUDA graphs)
- `async_scheduling: false`
- No streaming support

### Model Components

| Component | Parameters | Role |
|-----------|-----------|------|
| **Talker** (`Qwen3TTSTalkerForConditionalGeneration`) | 20 layers, 1024 hidden, 2048 intermediate | Text embeddings → first codebook logits (autoregressive) |
| **Code Predictor** (`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`) | 5 layers, 1024 hidden, 31 separate lm_heads | First codebook → remaining 31 codebooks (31 sequential steps) |
| **Speaker Encoder** (`Qwen3TTSSpeakerEncoder`) | ECAPA-TDNN, 1024-dim output | Reference audio → speaker embedding (Base task only) |
| **Speech Tokenizer** (`Qwen3TTSTokenizerV2Model`) | Mimi encoder + RVQ decoder + HiFi-GAN vocoder | Codec codes → waveform (non-autoregressive) |

### Token/Audio Mapping (12Hz variant)

- Frame rate: 12.5 Hz (1920 samples per code frame at 24 kHz)
- Quantizers: 16 (1 semantic + 15 acoustic), codebook size 2048
- 1 second of audio = ~12 codec frames, each with 16 discrete codes

## Bottleneck Analysis

### 1. Code Predictor Sequential Loop (Critical)

For each generated token in the talker's autoregressive loop, the code predictor runs **31 sequential forward passes** through its 5-layer transformer to produce codebooks 2-32. Each pass depends on the previous codebook's sampled output.

For T generated tokens: `T × 31 × forward(5-layer transformer)` = dominant cost.

A 10-second utterance at 12.5 Hz = 125 tokens × 31 passes = **3,875 forward passes** through the code predictor alone.

### 2. No Batching

`max_batch_size: 1` means requests are processed serially. Under concurrent load, requests queue and throughput scales linearly with latency.

### 3. No Streaming

The `/v1/audio/speech` endpoint generates the complete waveform before returning. The speech tokenizer decode happens after all codec codes are generated. There is no incremental audio delivery.

### 4. No CUDA Graphs

`enforce_eager: true` prevents CUDA graph capture, adding kernel launch overhead on every forward pass — amplified by the 31× code predictor loop.

### 5. Monolithic Forward Pass

Text encoding, codec generation, and waveform synthesis are fused into a single forward call. This prevents:
- Overlapping text encoding of request N+1 with waveform synthesis of request N
- Independent scaling of compute-heavy vs memory-heavy stages
- Stage-specific batch size tuning

## Proposed Optimizations

### Phase 1: Code Predictor Optimization

**Goal**: Reduce the 31× sequential bottleneck without changing the model architecture.

#### 1a. CUDA Graph Capture for Code Predictor

The code predictor has a fixed computation graph per step (5-layer transformer, fixed shapes during generation). Capture CUDA graphs for the inner loop:

```
enforce_eager: false  (for code predictor inner loop)
```

This eliminates kernel launch overhead across all 31 iterations. Expected improvement: 15-30% latency reduction on the code predictor loop, depending on GPU.

**Files to modify**:
- `vllm_omni/model_executor/stage_configs/qwen3_tts.yaml` — set `enforce_eager: false`
- `vllm_omni/model_executor/models/qwen3_tts/modeling_qwen3_tts.py` — ensure code predictor forward is graph-capturable (static shapes, no dynamic control flow)

#### 1b. KV Cache for Code Predictor

The code predictor currently recomputes attention over the full sequence on each of its 31 steps. Implementing KV caching within the code predictor's inner loop avoids redundant recomputation:

- Step 0: Full prefill of code predictor (sequence length = talker output length)
- Steps 1-30: Incremental decode with cached KV from previous steps

**Files to modify**:
- `vllm_omni/model_executor/models/qwen3_tts/modeling_qwen3_tts.py` — add `past_key_values` threading through `Qwen3TTSTalkerCodePredictorModel.forward()`

#### 1c. Parallel Codebook Prediction (Speculative)

Investigate whether codebooks 2-32 can be predicted in parallel rather than sequentially. This requires analysis of:
- How much quality degrades if codebooks are predicted independently
- Whether a distilled parallel predictor can match sequential quality
- Feasibility of predicting groups of codebooks in parallel (e.g., 4 groups of 8)

This is research-grade work and may not be feasible without retraining. Flag as speculative.

### Phase 2: Multi-Stage Decomposition

**Goal**: Decompose the monolithic TTS pipeline into separate stages, mirroring Qwen3 Omni's architecture.

#### Proposed 3-Stage Pipeline

```
Stage 0 (Talker):     Text → RVQ codec codes (32 codebooks)
                       Type: autoregressive LLM
                       Batch: up to 32
                       GPU: device 0

Stage 1 (Tokenizer Decoder): Codec codes → waveform
                       Type: non-autoregressive generation
                       Batch: up to 4
                       GPU: device 0

[Optional] Stage 2 (Speaker Encoder): Reference audio → speaker embedding
                       Type: non-autoregressive generation
                       Batch: up to 16
                       GPU: device 0
```

**Why 2 main stages instead of 3 like Omni**: Omni's thinker is a large multimodal LLM that benefits from tensor parallelism. TTS's talker is ~1.7B parameters and fits on a single GPU. The speaker encoder is lightweight and only needed for the Base task — it can remain fused with the talker or be a small optional stage.

#### Stage Config

```yaml
# qwen3_tts_multistage.yaml
stages:
  - stage_id: 0
    model_stage: qwen3_tts_talker
    model_arch: Qwen3TTSTalkerForConditionalGeneration
    stage_type: llm
    engine_output_type: audio_codes
    max_batch_size: 32
    gpu_memory_utilization: 0.7
    devices: "0"
    enforce_eager: false

  - stage_id: 1
    model_stage: qwen3_tts_decoder
    model_arch: Qwen3TTSTokenizerV2Decoder
    stage_type: generation
    engine_output_type: audio
    max_batch_size: 4
    gpu_memory_utilization: 0.2
    devices: "0"
    enforce_eager: true
```

#### Stage Input Processor

New file: `vllm_omni/model_executor/stage_input_processors/qwen3_tts.py`

```python
def talker2decoder(pooling_output, request):
    """Transfer codec codes from talker to tokenizer decoder."""
    return {
        "audio_codes": pooling_output["audio_codes"],
        "finished": torch.tensor(request.is_finished(), dtype=torch.bool),
    }
```

#### Required Changes

**New files**:
- `vllm_omni/model_executor/stage_input_processors/qwen3_tts.py`
- `vllm_omni/model_executor/stage_configs/qwen3_tts_multistage.yaml`

**Modified files**:
- `vllm_omni/model_executor/models/qwen3_tts/modeling_qwen3_tts.py` — split `Qwen3TTSForConditionalGeneration` so the talker and decoder can be loaded as separate stage models
- `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts.py` — update wrapper to support multi-stage mode
- `vllm_omni/model_executor/models/registry.py` — register talker and decoder as separate model architectures

### Phase 3: Streaming Audio Output

**Goal**: Deliver audio chunks incrementally as they are generated.

#### 3a. Async Chunk Pipeline

Enable `async_chunk: true` for the multi-stage TTS pipeline. This allows:
- Talker generates codec codes incrementally
- Decoder starts synthesizing waveform for completed chunks while talker continues
- Audio chunks delivered to client as they become available

This directly follows the pattern established by `qwen3_omni_moe_async_chunk.yaml`.

#### 3b. Speech Endpoint Streaming

Extend `/v1/audio/speech` to support streaming responses:

- Return chunked audio via `StreamingResponse`
- Each chunk is a valid audio segment (PCM frames or self-contained WAV chunks)
- Client receives audio while generation continues

**Files to modify**:
- `vllm_omni/entrypoints/openai/serving_speech.py` — add streaming path in `create_speech()`
- `vllm_omni/entrypoints/openai/protocol/audio.py` — remove SSE validation block, support `stream: true`
- `vllm_omni/entrypoints/openai/audio_utils_mixin.py` — add chunked audio encoding

#### 3c. Chunked Decode in Tokenizer

The 12Hz tokenizer decoder already supports chunked decode (300-frame chunks with 25-frame context overlap). Add a streaming variant with smaller chunks (~25 frames) for lower latency, similar to `qwen3_omni_code2wav.py`'s `chunked_decode_streaming()`.

### Phase 4: Batching and Throughput

**Goal**: Process multiple TTS requests concurrently.

#### 4a. Increase Batch Size

With multi-stage decomposition (Phase 2), the talker stage can batch multiple requests:
- Pad text sequences to max length in batch
- Run attention with proper masking
- Each request generates codec codes independently

The decoder stage can also batch, though with diminishing returns due to variable-length outputs.

#### 4b. Continuous Batching

Leverage vLLM's existing continuous batching for the talker stage. Requests enter and exit the batch dynamically as they complete, avoiding the "convoy effect" where short requests wait for long ones.

This requires the talker to be registered as a proper autoregressive LLM stage with vLLM's scheduler managing token budgets.

## Implementation Order

| Phase | Priority | Complexity | Expected Impact |
|-------|----------|-----------|-----------------|
| **1a** CUDA graphs | High | Low | 15-30% latency reduction |
| **1b** Code predictor KV cache | High | Medium | 20-40% latency reduction on code predictor |
| **2** Multi-stage decomposition | High | High | Enables phases 3 and 4; structural prerequisite |
| **3a** Async chunk pipeline | Medium | Medium | First-chunk latency reduction for streaming |
| **3b** Speech endpoint streaming | Medium | Medium | User-facing streaming support |
| **4a** Batch size increase | Medium | Low | Linear throughput scaling with batch size |
| **4b** Continuous batching | Low | Medium | Throughput under concurrent load |
| **1c** Parallel codebook prediction | Low | Research | Potentially 5-10× code predictor speedup (speculative) |

Phases 1a and 1b can proceed independently and in parallel. Phase 2 is the structural prerequisite for phases 3 and 4.

## Risks and Open Questions

1. **CUDA graph compatibility**: The code predictor's 31-step loop with dynamic sampling may not be fully graph-capturable. Need to verify that sampling operations (top-k, nucleus) can be captured or must remain eager.

2. **Quality regression from chunked streaming**: Smaller decode chunks may introduce boundary artifacts. Need perceptual evaluation (MOS testing) to validate chunk size choices.

3. **KV cache memory overhead**: Adding KV caching to the code predictor increases memory per request. With 5 layers and 31 steps, this is manageable but should be profiled.

4. **Multi-stage overhead**: Inter-stage communication adds latency. For a single request, multi-stage may be slower than monolithic. The benefit is throughput under load.

5. **Model weight splitting**: Decomposing `Qwen3TTSForConditionalGeneration` into separate stage models requires careful weight mapping to ensure pretrained weights load correctly into each stage.

6. **Parallel codebook feasibility**: Codebooks 2-32 may have strong sequential dependencies that make parallel prediction infeasible without retraining. This needs empirical investigation.
