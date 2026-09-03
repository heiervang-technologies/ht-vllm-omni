# Ruby runtime audit — raw measurements

Host: `ruby` (live-serving; no GPU benchmark performed)
Source baseline: `ht` at `db0e4590`

## Import-time attempt

Command:

```bash
TIMEFORMAT=$'wall_seconds=%R\nuser_seconds=%U\nsys_seconds=%S'
time env PYTHONPATH=. python -X importtime \
  -m vllm_omni.entrypoints.cli.main serve --help
```

Terminal result:

```text
ModuleNotFoundError: No module named 'aenum'
wall_seconds=0.048
user_seconds=0.037
sys_seconds=0.011
```

State: **could not measure serving import time**. Ruby has Python 3.14.5 and no
serving dependency environment; the project requires Python `<3.14`.

## Source and wheel payload

Commands:

```bash
git ls-tree -r -l ht | awk '{s+=$4} END {print s}'
du -sb .git
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv build --wheel
stat -c %s vllm_omni-0.0.0-py3-none-any.whl
unzip -l vllm_omni-0.0.0-py3-none-any.whl | tail -n 1
```

Raw values:

```text
tracked checkout bytes: 28203657
.git bytes: 37442872
wheel compressed bytes: 2624569
wheel member bytes: 9512094 (671 files)
```

Wheel contents explicitly checked:

```text
2727    vllm_omni/deploy/qwen3_tts.yaml
298880  vllm_omni/model_executor/models/covo_audio/speaker_prompt/prompt_latent.npy
2019    vllm_omni/platforms/npu/stage_configs/voxcpm.yaml
```

## Optional 25 Hz runtime tail

Commands used `pip download --no-deps --only-binary=:all:` for CPython 3.12,
manylinux x86-64, followed by `stat` and `unzip -l`.

```text
onnxruntime 1.23.2 wheel compressed bytes: 17382649
onnxruntime 1.23.2 wheel member bytes: 49610946 (351 files)
onnxruntime 1.28.0 wheel compressed bytes: 19214257
onnxruntime 1.28.0 wheel member bytes: 54323426 (353 files)
```

These are package payload measurements, not claimed Docker-image savings; base
image overlap and filesystem layer compression must be measured in the built
image.

## Artifact baseline access

```text
docker buildx imagetools inspect ghcr.io/heiervang-technologies/ht-vllm-omni:ht
ERROR: failed to authorize ... 401 Unauthorized
```

State: **could not independently reproduce the mission-provided ~19.5 GB image
size on ruby**. No image was pulled and the live worker was not disturbed.
