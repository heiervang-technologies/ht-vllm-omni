from pathlib import Path

_ROOT = Path(__file__).parents[2]


def _requirements(path: Path) -> list[str]:
    return [
        line
        for raw_line in path.read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


def test_qwen3_tts_12hz_profile_excludes_optional_runtime_stacks():
    requirements = _requirements(_ROOT / "requirements/qwen3_tts_12hz.txt")
    excluded = (
        "av",
        "diffusers",
        "accelerate",
        "cache-dit",
        "torchsde",
        "openai-whisper",
        "imageio",
        "x-transformers",
        "onnxruntime",
        "fa3-fwd",
    )

    assert requirements
    assert not any(requirement.startswith(excluded) for requirement in requirements)


def test_slim_image_bakes_selected_profile_into_wheel_metadata():
    dockerfile = (_ROOT / "docker/Dockerfile.slim").read_text()
    setup = (_ROOT / "setup.py").read_text()

    assert "VLLM_OMNI_INSTALL_PROFILE=${VLLM_OMNI_INSTALL_PROFILE}" in dockerfile
    assert '"qwen3_tts_12hz": requirements_dir / "qwen3_tts_12hz.txt"' in setup
