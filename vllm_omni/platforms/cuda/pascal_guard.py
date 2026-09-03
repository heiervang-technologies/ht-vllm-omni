"""Fail-loud CUDA runtime checks for Pascal (sm_61) deployments."""

from __future__ import annotations

import re
from collections.abc import Iterable


def _cuda_release(version: str | None) -> tuple[int, int] | None:
    if version is None:
        return None
    match = re.match(r"^(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def validate_pascal_runtime(
    *,
    device_capability: tuple[int, int],
    cuda_version: str | None,
    compiled_arches: Iterable[str],
    dtype_name: str | None,
    checkpoint_dtype_name: str | None = None,
) -> None:
    """Reject wheels and dtypes that cannot execute correctly on sm_61."""
    if device_capability != (6, 1):
        return

    release = _cuda_release(cuda_version)
    if release is None or release > (12, 6):
        rendered = cuda_version or "none"
        raise RuntimeError(
            "Pascal sm_61 requires a PyTorch CUDA 12.6-or-earlier build; "
            f"this wheel reports CUDA {rendered}. Install a cu126 wheel or an sm_61 source build."
        )

    arches = set(compiled_arches)
    if not ({"sm_61", "compute_61"} & arches):
        rendered = ", ".join(sorted(arches)) or "none"
        raise RuntimeError(
            "PyTorch was built without Pascal kernels (sm_61); "
            f"compiled architectures: {rendered}. Install a cu126 wheel or rebuild with TORCH_CUDA_ARCH_LIST=6.1."
        )

    if dtype_name is not None and ("bfloat16" in dtype_name.lower() or "bf16" in dtype_name.lower()):
        raise RuntimeError(
            "bfloat16 is unsupported on Pascal sm_61; use checkpoint-compatible float32 or validated INT8."
        )

    requested_fp16 = dtype_name is not None and ("float16" in dtype_name.lower() or "fp16" in dtype_name.lower())
    checkpoint_bf16 = checkpoint_dtype_name is not None and (
        "bfloat16" in checkpoint_dtype_name.lower() or "bf16" in checkpoint_dtype_name.lower()
    )
    if requested_fp16 and checkpoint_bf16:
        raise RuntimeError(
            "The checkpoint declares bfloat16 weights; casting it to float16 on Pascal can overflow and corrupt ASR "
            "output. Use float32 or a validated INT8 checkpoint."
        )
