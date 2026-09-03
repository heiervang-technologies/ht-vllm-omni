import ast
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).parents[2]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_scope_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_diffusion_data_does_not_import_diffusers_at_runtime():
    imports = _module_scope_imports(_ROOT / "vllm_omni/diffusion/data.py")

    assert "diffusers" not in imports


def test_serve_defers_api_server_until_command_execution():
    imports = _module_scope_imports(_ROOT / "vllm_omni/entrypoints/cli/serve.py")

    assert "vllm_omni.entrypoints.openai.api_server" not in imports


def test_cli_package_does_not_import_benchmark_dependencies_at_runtime():
    imports = _module_scope_imports(_ROOT / "vllm_omni/entrypoints/cli/__init__.py")

    assert "vllm_omni.benchmarks.patch" not in imports
    assert "vllm_omni.entrypoints.cli.benchmark.serve" not in imports


def test_cli_selects_first_positional_command_past_global_flags():
    cli_main = _load_module(_ROOT / "vllm_omni/entrypoints/cli/main.py", "_cli_main_under_test")

    assert cli_main._first_positional_argument(["vllm-omni", "serve", "model", "--omni"]) == "serve"
    assert cli_main._first_positional_argument(["vllm-omni", "--omni", "bench", "serve"]) == "bench"
    assert cli_main._first_positional_argument(["vllm-omni", "--omni", "--version"]) is None
