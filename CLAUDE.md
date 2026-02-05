# ht-vllm-omni

Fork of [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) for Qwen3 TTS optimization work.

## Build Rules

### flash-attn: NEVER build from source

**Do NOT compile flash-attn.** Building flash-attn from source takes 30+ minutes, requires matching CUDA toolkit versions, and frequently fails. Instead:

- **Install pre-built wheels only**: `pip install flash-attn --no-build-isolation` or use the pre-built wheels from the [flash-attn releases](https://github.com/Dao-AILab/flash-attention/releases).
- **In Docker**: The base image `vllm/vllm-openai` may already include flash-attn. Check first with `python -c "import flash_attn"`. If not present, install a pre-built wheel matching the CUDA version.
- **Locally (no GPU)**: Do not attempt to install flash-attn at all. The SDPA fallback handles this.

This applies to all contexts: Dockerfiles, CI scripts, pip install commands, and sub-agent tasks.

## Development

### Running tests locally (no vllm needed)

The Qwen3 TTS regression tests use compile+exec to bypass the heavy `vllm_omni.__init__` import chain:

```bash
uv run python -m pytest tests/model_executor/models/qwen3_tts/ -v --noconftest -o "addopts="
```

### Running tests in Docker

```bash
docker run --rm vllm-omni-q3tts:latest python -m pytest tests/model_executor/models/qwen3_tts/ -v
```

### Docker build

```bash
docker build -t vllm-omni-q3tts:latest -f Dockerfile.ci .
```

### Docker run (inference server)

```bash
docker run -d \
  --name qwen3-tts-server \
  --gpus '"device=0"' \
  --cpus=4 --memory=48g --shm-size=4g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  vllm-omni-q3tts:latest \
  vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --omni --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.9 --enforce-eager
```

## Branches

- `main` — upstream tracking
- `q3-tts` — Qwen3 TTS integration (our working base)
- `feat/*` — feature branches, PR against `q3-tts`
