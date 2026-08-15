import ast
from pathlib import Path


def test_tokenizer_modeling_modules_are_function_local_imports():
    """The wrapper import must not pull both tokenizer graphs into every worker."""
    source_path = (
        Path(__file__).parents[4]
        / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_tokenizer.py"
    )
    tree = ast.parse(source_path.read_text())

    eager_modeling_imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and ".modeling_qwen3_tts_tokenizer_" in node.module:
            eager_modeling_imports.append(node.module)

    assert eager_modeling_imports == []


def test_talker_does_not_force_bfloat16_over_engine_dtype():
    source_path = Path(__file__).parents[4] / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py"

    assert "torch.bfloat16" not in source_path.read_text()
