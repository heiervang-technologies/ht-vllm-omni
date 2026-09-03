# SPDX-License-Identifier: Apache-2.0
"""Fail-loud CUDA runtime checks for legacy GPU deployments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_PASCAL_CAPABILITY = (6, 1)
_PASCAL_MAX_CUDNN = (9, 11, 1)
_REQUIRED_ARCH_ENV = "VLLM_OMNI_REQUIRED_CUDA_ARCH"


@dataclass(frozen=True)
class CudaRuntimeReport:
    device_capability: tuple[int, int]
    compiled_arches: tuple[str, ...]
    cuda_version: str
    cudnn_version: tuple[int, int, int] | None


def _normalize_arch(arch: str) -> str:
    value = arch.strip().lower().replace("compute_", "sm_")
    if value.startswith("sm") and not value.startswith("sm_"):
        value = f"sm_{value[2:]}"
    if value.replace(".", "", 1).isdigit():
        value = f"sm_{value.replace('.', '')}"
    return value


def _decode_cudnn_version(version: int | None) -> tuple[int, int, int] | None:
    if version is None:
        return None
    return version // 10000, (version % 10000) // 100, version % 100


def validate_cuda_runtime(torch_module: Any | None = None) -> CudaRuntimeReport | None:
    """Validate CUDA binaries before a legacy GPU reaches model loading.

    Pascal sm_61 is detected automatically. ``VLLM_OMNI_REQUIRED_CUDA_ARCH``
    can force the same check in a container/CI job where the target GPU is not
    visible (for example, ``sm_61`` or ``6.1``).
    """
    if torch_module is None:
        import torch as torch_module

    required_from_env = os.environ.get(_REQUIRED_ARCH_ENV)
    cuda = getattr(torch_module, "cuda", None)
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if cuda is None or not cuda.is_available():
        if required_from_env:
            raise RuntimeError(
                f"CUDA compatibility preflight requires {_REQUIRED_ARCH_ENV}={required_from_env}, "
                "but torch.cuda is unavailable."
            )
        return None

    capability = tuple(int(part) for part in cuda.get_device_capability())
    required_arch = _normalize_arch(required_from_env) if required_from_env else None
    if capability == _PASCAL_CAPABILITY:
        required_arch = "sm_61"
    if required_arch is None:
        return None

    compiled_arches = tuple(str(arch).lower() for arch in cuda.get_arch_list())
    normalized_arches = {_normalize_arch(arch) for arch in compiled_arches}
    cudnn_raw = torch_module.backends.cudnn.version()
    cudnn_version = _decode_cudnn_version(cudnn_raw)

    errors: list[str] = []
    actual_arch = f"sm_{capability[0]}{capability[1]}"
    if actual_arch != required_arch:
        errors.append(f"visible GPU is {actual_arch}, required {required_arch}")
    if required_arch not in normalized_arches:
        errors.append(f"torch wheel has no {required_arch} kernels (compiled arches: {list(compiled_arches)!r})")
    if cuda_version is None:
        errors.append("torch is not a CUDA build")
    elif required_arch == "sm_61" and int(cuda_version.split(".", 1)[0]) >= 13:
        errors.append(f"CUDA {cuda_version} cannot target Pascal sm_61; use a CUDA 12.x build")
    if required_arch == "sm_61" and cudnn_version is not None and cudnn_version > _PASCAL_MAX_CUDNN:
        errors.append(
            "cuDNN "
            f"{'.'.join(map(str, cudnn_version))} exceeds the Pascal ceiling "
            f"{'.'.join(map(str, _PASCAL_MAX_CUDNN))}"
        )

    if errors:
        raise RuntimeError("CUDA compatibility preflight failed: " + "; ".join(errors))

    return CudaRuntimeReport(
        device_capability=capability,
        compiled_arches=compiled_arches,
        cuda_version=str(cuda_version),
        cudnn_version=cudnn_version,
    )
