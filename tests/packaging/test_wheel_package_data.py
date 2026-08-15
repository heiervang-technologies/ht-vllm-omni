from pathlib import Path


def test_runtime_yaml_and_model_assets_are_declared_as_package_data():
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text()

    assert '"*" = ["*.yaml", "*.npy", "*.npz"]' in pyproject
