# Qwen3 TTS 0.6B Base - Issues Log

Tracking all issues encountered while getting `Qwen/Qwen3-TTS-12Hz-0.6B-Base` running on local RTX 3090 hardware via vLLM-Omni Docker.

## Environment

- **GPU**: 2x NVIDIA RTX 3090 (24GB VRAM each, Ampere sm_86)
- **Host OS**: Arch Linux (Omarchy)
- **Docker image**: `vllm-omni-q3tts:latest` (based on `vllm/vllm-openai:v0.14.0`)
- **vLLM-Omni version**: 0.14.0rc1
- **Model**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- **Branch**: `q3-tts`

## Resource Constraints Applied

```bash
docker run \
  --cpus=4 \
  --memory=48g \
  --shm-size=4g \
  --gpus '"device=0"' \
  -e OMP_NUM_THREADS=4 \
  -e MKL_NUM_THREADS=4 \
  -e NUMEXPR_MAX_THREADS=4
```

---

## Issue 1: Profile Run Hang (Shared Memory Broadcast Timeout)

**Status**: Fixed locally
**Severity**: Blocker - server never starts
**Upstream issue**: #995

### Symptoms

Server hangs during startup with:
```
Setting `pad_token_id` to `eos_token_id`:2150 for open-end generation.
[shm_broadcast.py:542] No available shared memory broadcast block found in 60 seconds.
```

### Root Cause

vLLM's profile/warmup run calls the model's `forward()` with dummy token IDs and empty `runtime_additional_information`. For the Base task type, this triggers `generate_voice_clone()` in ICL mode with:
- A 1-second silent audio clip as ref_audio (all zeros)
- Placeholder text "For profile run" as ref_text
- `x_vector_only_mode=False` (default) forcing ICL mode

The 0.6B model cannot converge from this degenerate input and generates indefinitely, never producing an EOS token. The EngineCore times out waiting for the worker to finish the profile run.

The 1.7B models are robust enough to produce EOS from the same degenerate input, which is why only 0.6B is affected.

### Fix Applied

Short-circuit in `forward()` when `text` is empty (profile run detection):

```python
# vllm_omni/model_executor/models/qwen3_tts/qwen3_tts.py, line 128
if not text:
    logger.info("Profile run detected (empty text). Returning dummy audio output.")
    dummy_audio = torch.zeros(24000, dtype=torch.float32)
    return OmniOutput(
        text_hidden_states=None,
        multimodal_outputs={
            "model_outputs": dummy_audio,
            "sr": torch.tensor(24000, dtype=torch.int),
        },
    )
```

### Files Modified
- `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts.py`

---

## Issue 2: SoX Not Installed in Docker Image

**Status**: Non-blocking warning
**Severity**: Low (cosmetic warning, not used in core pipeline)

### Symptoms

```
/bin/sh: 1: sox: not found
WARNING __init__.py:10: SoX could not be found!
```

### Root Cause

The `sox` Python package is installed but the system `sox` binary is not. The `Dockerfile.ci` only installs `ffmpeg`. The ROCm Dockerfile (`Dockerfile.rocm`) does install `sox` via pip + installs `onnxruntime-rocm`, but the standard CI Dockerfile doesn't install the system binary.

### Impact

The `sox` Python wrapper requires the system binary for audio manipulation. Currently the TTS pipeline uses `soundfile` and `librosa` for audio I/O, so this doesn't block inference. It could affect some audio preprocessing operations if they rely on SoX.

### Fix (if needed)

Add to `Dockerfile.ci`:
```dockerfile
RUN apt-get update && apt-get install -y ffmpeg sox && ...
```

---

## Issue 3: Flash Attention Not Available

**Status**: Performance degradation
**Severity**: Medium (inference works but slower)

### Symptoms

```
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
[qwen3_tts.py:76] Flash-Attn is not installed. Using default PyTorch attention implementation.
```

### Root Cause

The base image `vllm/vllm-openai:v0.14.0` includes vLLM's custom attention kernels but not the standalone `flash-attn` package that the Qwen3 TTS model's HuggingFace implementation expects. The TTS model code (`modeling_qwen3_tts.py`) checks for `flash_attn` independently of vLLM's attention backends.

### Impact

Falls back to PyTorch's native attention implementation which is significantly slower, especially for longer sequences. The code predictor runs 31 sequential attention passes per generated token — this slowdown compounds.

### Fix Options

1. Install `flash-attn` in the Docker image (requires CUDA compilation during build, slow)
2. Patch the model to use vLLM's built-in attention instead of importing `flash_attn` directly

---

## Issue 4: Mooncake/Yuanrong Connector Warnings

**Status**: Non-blocking
**Severity**: Informational

### Symptoms

```
WARNING [mooncake_connector.py:18] Mooncake not available, MooncakeOmniConnector will not work
WARNING [yuanrong_connector.py:14] Datasystem not available, YuanrongConnector will not work
```

### Root Cause

These are optional distributed connectors for specific deployment environments (Mooncake for shared memory, Yuanrong for data system). Not needed for local single-node inference.

### Impact

None. These warnings appear during initialization and can be safely ignored.

---

## Issue 5: vllm_config is None Warning

**Status**: Non-blocking
**Severity**: Low

### Symptoms

```
WARNING [api_server.py:388] vllm_config is None, some features may not work correctly
WARNING [api_server.py:471] Cannot initialize processors: vllm_config is None
```

### Root Cause

The OmniServing layer constructs the API server without passing `vllm_config` to some components. This is a known architectural gap where the multi-stage engine doesn't expose the config to the API layer the same way single-model vLLM does.

### Impact

Some API features that depend on `vllm_config` (like tokenizer-based preprocessing) may not work. TTS inference itself is not affected since it uses its own internal tokenizer.

---

## Issue 6: max_num_batched_tokens Warning

**Status**: Non-blocking
**Severity**: Low

### Symptoms

```
WARNING [scheduler.py:271] max_num_batched_tokens (1000000) exceeds max_num_seqs * max_model_len (32768)
```

### Root Cause

The TTS stage config sets `max_num_batched_tokens: 1000000` (a sentinel value for "unlimited") but `max_num_seqs: 1` with `max_model_len: 32768`. The scheduler warns about the mismatch but operates correctly with `max_num_seqs=1`.

### Impact

None. The scheduler correctly limits to 1 sequence at a time regardless.

---

## Issue 7: Generation Config Override Warning

**Status**: Non-blocking
**Severity**: Low

### Symptoms

```
WARNING [model.py:1358] Default sampling parameters have been overridden by the model's
Hugging Face generation config recommended from the model creator.
```

### Root Cause

The HuggingFace model config includes generation parameters (temperature, top_k, etc.) that override vLLM's defaults. This is intentional — the model creator tuned these for best TTS quality.

### Impact

None negative. The model uses its own recommended sampling parameters.

---

## Working Configuration

After fixing Issue 1, the server starts and serves requests successfully:

```bash
docker run -d \
  --name qwen3-tts-server \
  --gpus '"device=0"' \
  --cpus=4 \
  --memory=48g \
  --shm-size=4g \
  -p 8000:8000 \
  -v /home/me/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e OMP_NUM_THREADS=4 \
  -e MKL_NUM_THREADS=4 \
  -e NUMEXPR_MAX_THREADS=4 \
  vllm-omni-q3tts:latest \
  vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --omni \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager
```

### Verified Working Request

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test.",
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "task_type": "Base",
    "language": "English",
    "ref_audio": "data:audio/wav;base64,<base64_encoded_wav>",
    "ref_text": "A reference audio sample.",
    "x_vector_only_mode": true,
    "max_new_tokens": 256
  }' -o output.wav
```

### Performance (First Request)

- Model load: 2.02 GiB, 6.28 seconds
- Inference: ~2.0 seconds for 1.82s of audio (0.91x realtime)
- 15 tokens generated at 7.5 tokens/s
- Output: 24kHz, mono, valid non-silent audio

### GPU Memory Usage

- Model weights: 2.02 GiB
- Available VRAM: 24 GiB (RTX 3090)
- `gpu-memory-utilization: 0.9` = up to 21.6 GiB usable
