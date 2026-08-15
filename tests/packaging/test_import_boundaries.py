import ast
from pathlib import Path

_ROOT = Path(__file__).parents[2]


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
