from pathlib import Path


def test_talker_does_not_force_bfloat16_over_engine_dtype():
    source_path = Path(__file__).parents[4] / "vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py"

    assert "torch.bfloat16" not in source_path.read_text()
