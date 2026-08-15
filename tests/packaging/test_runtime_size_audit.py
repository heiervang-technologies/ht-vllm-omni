import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).parents[2] / "tools/runtime_size_audit.py"
_SPEC = importlib.util.spec_from_file_location("_runtime_size_audit_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_logical_size_does_not_follow_directory_symlinks(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "payload").write_bytes(b"12345")
    (data / "loop").symlink_to(data, target_is_directory=True)

    assert _MODULE._logical_size(data) == 5 + (data / "loop").lstat().st_size


def test_build_report_limits_sorted_distributions(monkeypatch):
    rows = [
        {"name": "small", "logical_bytes": 1},
        {"name": "large", "logical_bytes": 10},
    ]
    monkeypatch.setattr(_MODULE, "_distribution_sizes", lambda: rows)
    monkeypatch.setattr(_MODULE, "_component_paths", lambda: [])

    report = _MODULE.build_report(top=1)

    assert report["distribution_count"] == 2
    assert report["top_distributions"] == [rows[0]]
