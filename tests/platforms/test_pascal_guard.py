import importlib.util
from pathlib import Path

import pytest


def _load_module():
    source = Path(__file__).parents[2] / "vllm_omni/platforms/cuda/pascal_guard.py"
    spec = importlib.util.spec_from_file_location("_test_pascal_guard", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_module()


def test_accepts_cu126_pascal_wheel_with_sm61() -> None:
    guard.validate_pascal_runtime(
        device_capability=(6, 1),
        cuda_version="12.6",
        compiled_arches=["sm_61", "sm_70"],
        dtype_name="torch.float16",
    )


@pytest.mark.parametrize("cuda_version", [None, "13.0", "12.8"])
def test_rejects_unsupported_pascal_cuda_wheel(cuda_version: str | None) -> None:
    with pytest.raises(RuntimeError, match="CUDA 12.6-or-earlier"):
        guard.validate_pascal_runtime(
            device_capability=(6, 1),
            cuda_version=cuda_version,
            compiled_arches=["sm_61"],
            dtype_name="torch.float16",
        )


def test_rejects_wheel_without_sm61_kernels() -> None:
    with pytest.raises(RuntimeError, match="without Pascal kernels"):
        guard.validate_pascal_runtime(
            device_capability=(6, 1),
            cuda_version="12.6",
            compiled_arches=["sm_80", "sm_90"],
            dtype_name="torch.float16",
        )


def test_rejects_bfloat16_on_pascal() -> None:
    with pytest.raises(RuntimeError, match="bfloat16 is unsupported"):
        guard.validate_pascal_runtime(
            device_capability=(6, 1),
            cuda_version="12.6",
            compiled_arches=["sm_61"],
            dtype_name="torch.bfloat16",
        )


def test_rejects_fp16_cast_of_bfloat16_checkpoint() -> None:
    with pytest.raises(RuntimeError, match="checkpoint declares bfloat16"):
        guard.validate_pascal_runtime(
            device_capability=(6, 1),
            cuda_version="12.6",
            compiled_arches=["sm_61"],
            dtype_name="torch.float16",
            checkpoint_dtype_name="bfloat16",
        )


def test_does_not_apply_pascal_policy_to_newer_gpu() -> None:
    guard.validate_pascal_runtime(
        device_capability=(8, 0),
        cuda_version="13.0",
        compiled_arches=["sm_80"],
        dtype_name="torch.bfloat16",
    )
