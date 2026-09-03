"""CLI helpers for vLLM-Omni entrypoints.

Commands are exposed lazily so importing the serving CLI does not also import
benchmark-only audio/evaluation dependencies.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

    from .serve import OmniServeCommand


def __getattr__(name: str):
    if name == "OmniServeCommand":
        from .serve import OmniServeCommand

        return OmniServeCommand
    if name == "OmniBenchmarkServingSubcommand":
        from vllm_omni.benchmarks.patch import patch as _benchmark_patch  # noqa: F401
        from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

        return OmniBenchmarkServingSubcommand
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OmniServeCommand", "OmniBenchmarkServingSubcommand"]
