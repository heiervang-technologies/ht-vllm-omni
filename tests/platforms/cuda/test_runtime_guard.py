import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SOURCE = Path(__file__).parents[3] / "vllm_omni/platforms/cuda/runtime_guard.py"
_SPEC = importlib.util.spec_from_file_location("_cuda_runtime_guard_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
validate_cuda_runtime = _MODULE.validate_cuda_runtime


class _FakeCuda:
    def __init__(self, *, available=True, capability=(6, 1), arches=("sm_61",)):
        self._available = available
        self._capability = capability
        self._arches = arches

    def is_available(self):
        return self._available

    def get_device_capability(self):
        return self._capability

    def get_arch_list(self):
        return list(self._arches)


def _fake_torch(*, available=True, capability=(6, 1), arches=("sm_61",), cuda_version="12.6", cudnn=91101):
    return SimpleNamespace(
        cuda=_FakeCuda(available=available, capability=capability, arches=arches),
        version=SimpleNamespace(cuda=cuda_version),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: cudnn)),
    )


def test_pascal_runtime_report_accepts_compatible_build(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_REQUIRED_CUDA_ARCH", raising=False)

    report = validate_cuda_runtime(_fake_torch())

    assert report is not None
    assert report.device_capability == (6, 1)
    assert report.compiled_arches == ("sm_61",)
    assert report.cuda_version == "12.6"
    assert report.cudnn_version == (9, 11, 1)


@pytest.mark.parametrize(
    ("torch_module", "expected"),
    [
        (_fake_torch(arches=("sm_80", "sm_90")), "no sm_61 kernels"),
        (_fake_torch(cuda_version="13.0"), "CUDA 13.0 cannot target Pascal"),
        (_fake_torch(cudnn=91200), "exceeds the Pascal ceiling 9.11.1"),
    ],
)
def test_pascal_runtime_rejects_incompatible_builds(monkeypatch, torch_module, expected):
    monkeypatch.delenv("VLLM_OMNI_REQUIRED_CUDA_ARCH", raising=False)

    with pytest.raises(RuntimeError, match=expected):
        validate_cuda_runtime(torch_module)


def test_non_pascal_cuda_is_unchanged_without_override(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_REQUIRED_CUDA_ARCH", raising=False)

    assert validate_cuda_runtime(_fake_torch(capability=(8, 0), arches=("sm_80",))) is None


def test_required_arch_override_checks_gpu_and_wheel(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_REQUIRED_CUDA_ARCH", "6.1")

    with pytest.raises(RuntimeError, match="visible GPU is sm_80, required sm_61"):
        validate_cuda_runtime(_fake_torch(capability=(8, 0), arches=("sm_61", "sm_80")))


def test_required_arch_override_fails_when_cuda_is_hidden(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_REQUIRED_CUDA_ARCH", "sm_61")

    with pytest.raises(RuntimeError, match="torch.cuda is unavailable"):
        validate_cuda_runtime(_fake_torch(available=False))
