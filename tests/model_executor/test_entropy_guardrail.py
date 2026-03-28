# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the entropy-based degeneration guardrail in Qwen3TTSTalker.

The guardrail method ``_apply_entropy_guardrail`` lives on
``Qwen3TTSTalkerForCausalLM`` which has a heavy import chain (vllm, librosa,
etc.).  To keep these tests lightweight and runnable without a full install we
build a thin shim that carries only the fields the method reads and bind the
method source directly via ``types.FunctionType``.

No GPU, no model weights, no server required.
"""

from __future__ import annotations

import ast
import logging
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 – the method body uses F

# ---------------------------------------------------------------------------
# Extract ``_apply_entropy_guardrail`` source from the talker module without
# importing it (which would pull in vllm, librosa, …).
# ---------------------------------------------------------------------------

_TALKER_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_omni"
    / "model_executor"
    / "models"
    / "qwen3_tts"
    / "qwen3_tts_talker.py"
)


def _extract_method(source_path: Path, method_name: str):
    """Parse the source file's AST and compile just the target method."""
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            # Wrap in a module so compile() is happy
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            code = compile(mod, str(source_path), "exec")
            ns: dict[str, Any] = {"torch": torch, "F": F, "logger": logging.getLogger("entropy_guardrail_test")}
            exec(code, ns)  # noqa: S102
            return ns[method_name]
    raise ValueError(f"{method_name} not found in {source_path}")


_guardrail_fn = _extract_method(_TALKER_PATH, "_apply_entropy_guardrail")

# ---------------------------------------------------------------------------
# Minimal shim that carries only the state _apply_entropy_guardrail touches.
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256  # large enough that uniform entropy (~5.5) exceeds default thresholds
EOS_ID = 0  # treat token-0 as codec EOS

logger = logging.getLogger("vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker")


class _GuardrailShim:
    """Mimics just the fields that ``_apply_entropy_guardrail`` reads."""

    def __init__(
        self,
        *,
        eos_id: int = EOS_ID,
        enabled: bool = True,
        threshold_high: float = 4.5,
        threshold_low: float = 0.5,
        window: int = 5,
    ):
        self._codec_eos_token_id = eos_id
        self._entropy_state: dict[int, dict[str, Any]] = {}
        self._entropy_defaults: dict[str, Any] = {
            "enabled": enabled,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "window": window,
        }
        self._entropy_config_staging: list[dict[str, Any]] = []
        self._entropy_guardrail_triggered: dict[int, dict[str, Any]] = {}

    def _apply_entropy_guardrail(self, logits, sampling_metadata):
        return _guardrail_fn(self, logits, sampling_metadata)


def _sampling_meta(output_token_ids: list[list[int]]):
    return types.SimpleNamespace(output_token_ids=output_token_ids)


def _uniform_logits(batch: int = 1, vocab: int = VOCAB_SIZE) -> torch.Tensor:
    """Logits that produce ~uniform distribution -> high entropy."""
    return torch.zeros(batch, vocab)


def _peaked_logits(
    batch: int = 1, vocab: int = VOCAB_SIZE, peak_id: int = 5
) -> torch.Tensor:
    """Logits that produce a very peaked distribution -> very low entropy."""
    logits = torch.full((batch, vocab), -20.0)
    logits[:, peak_id] = 20.0
    return logits


def _normal_logits(batch: int = 1, vocab: int = VOCAB_SIZE) -> torch.Tensor:
    """Logits with moderate entropy (within default thresholds ~1-3 nats)."""
    # Create a distribution that's peaked enough to keep entropy in [1, 3.5]
    logits = torch.full((batch, vocab), -10.0)
    # Spread probability mass over ~10 tokens
    for i in range(10):
        logits[:, i] = torch.randn(batch) + 2.0
    return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntropyGuardrailBasics:
    """Core behaviour: trigger on sustained OOB, recover when in-bounds."""

    def test_disabled_by_default_is_noop(self):
        shim = _GuardrailShim(enabled=False)
        tokens = [list(range(10))]
        logits = _uniform_logits()
        meta = _sampling_meta(tokens)
        result = shim._apply_entropy_guardrail(logits, meta)
        assert torch.equal(result, logits), "Disabled guardrail should not modify logits"

    def test_early_steps_skipped(self):
        """Steps < 2 should be skipped (no meaningful entropy)."""
        shim = _GuardrailShim(enabled=True, window=1)
        tokens = [[42]]  # only 1 token so far
        logits = _uniform_logits()
        meta = _sampling_meta(tokens)
        result = shim._apply_entropy_guardrail(logits, meta)
        assert torch.equal(result, logits), "Step < 2 should be skipped"

    def test_no_output_token_ids_is_noop(self):
        shim = _GuardrailShim(enabled=True)
        logits = _uniform_logits()
        meta = types.SimpleNamespace(output_token_ids=None)
        result = shim._apply_entropy_guardrail(logits, meta)
        assert torch.equal(result, logits)

    def test_missing_attr_is_noop(self):
        shim = _GuardrailShim(enabled=True)
        logits = _uniform_logits()
        meta = types.SimpleNamespace()
        result = shim._apply_entropy_guardrail(logits, meta)
        assert torch.equal(result, logits)


class TestHighEntropyTrigger:
    """Sustained high entropy (uniform logits) should force EOS."""

    def test_forces_eos_after_window_exceeded(self):
        window = 3
        shim = _GuardrailShim(enabled=True, threshold_high=4.0, window=window)
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        triggered = False
        for step in range(window + 2):
            tokens[0].append(99)
            logits = _uniform_logits()
            result = shim._apply_entropy_guardrail(logits, meta)
            if result[0, EOS_ID] == 100.0:
                triggered = True
                break

        assert triggered, "EOS should have been forced after window consecutive OOB steps"
        for tok_id in range(VOCAB_SIZE):
            if tok_id != EOS_ID:
                assert result[0, tok_id] == float("-inf"), (
                    f"Non-EOS token {tok_id} should be -inf"
                )

    def test_resets_after_trigger(self):
        """After triggering, the consecutive counter resets so it doesn't
        fire again immediately on the very next step."""
        window = 2
        shim = _GuardrailShim(enabled=True, threshold_high=4.0, window=window)
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        # Trigger
        for _ in range(window + 1):
            tokens[0].append(99)
            shim._apply_entropy_guardrail(_uniform_logits(), meta)

        # Next step with normal logits should NOT trigger
        tokens[0].append(99)
        logits = _normal_logits()
        result = shim._apply_entropy_guardrail(logits, meta)
        assert result[0, EOS_ID] != 100.0, "Should not trigger right after reset"


class TestLowEntropyTrigger:
    """Sustained low entropy (peaked logits) should also force EOS."""

    def test_forces_eos_on_collapsed_entropy(self):
        window = 3
        shim = _GuardrailShim(
            enabled=True, threshold_low=0.5, threshold_high=10.0, window=window
        )
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        triggered = False
        for _ in range(window + 2):
            tokens[0].append(99)
            logits = _peaked_logits()
            result = shim._apply_entropy_guardrail(logits, meta)
            if result[0, EOS_ID] == 100.0:
                triggered = True
                break

        assert triggered, "Collapsed entropy should force EOS"


class TestRecoveryWithinWindow:
    """If entropy returns to normal before the window is exhausted, no trigger."""

    def test_no_trigger_when_recovered(self):
        window = 5
        shim = _GuardrailShim(enabled=True, threshold_high=4.0, window=window)
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        # 3 OOB steps (< window)
        for _ in range(3):
            tokens[0].append(99)
            shim._apply_entropy_guardrail(_uniform_logits(), meta)

        # Then recover with normal entropy
        tokens[0].append(99)
        result = shim._apply_entropy_guardrail(_normal_logits(), meta)

        assert result[0, EOS_ID] != 100.0, "Should not trigger after recovery"
        key = id(tokens[0])
        assert shim._entropy_state[key]["consecutive_oob"] == 0


class TestStagedConfig:
    """Per-request config staging (from preprocess) is consumed correctly."""

    def test_staged_config_overrides_defaults(self):
        shim = _GuardrailShim(enabled=False)
        shim._entropy_config_staging = [
            {"enabled": True, "threshold_high": 3.0, "threshold_low": 0.1, "window": 2}
        ]
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        shim._apply_entropy_guardrail(_uniform_logits(), meta)

        key = id(tokens[0])
        assert shim._entropy_state[key]["cfg"]["enabled"] is True
        assert shim._entropy_state[key]["cfg"]["threshold_high"] == 3.0

    def test_staged_config_consumed_in_order(self):
        shim = _GuardrailShim(enabled=False)
        shim._entropy_config_staging = [
            {"enabled": True, "threshold_high": 3.0, "threshold_low": 0.1, "window": 2},
            {"enabled": False, "threshold_high": 5.0, "threshold_low": 0.5, "window": 5},
        ]
        tokens_a = list(range(10))
        tokens_b = list(range(10))
        meta = _sampling_meta([tokens_a, tokens_b])

        shim._apply_entropy_guardrail(_uniform_logits(batch=2), meta)

        assert shim._entropy_state[id(tokens_a)]["cfg"]["enabled"] is True
        assert shim._entropy_state[id(tokens_b)]["cfg"]["enabled"] is False


class TestBatchHandling:
    """Guardrail handles batches with mixed enabled/disabled requests."""

    def test_only_enabled_request_triggers(self):
        shim = _GuardrailShim(enabled=False)
        shim._entropy_config_staging = [
            {"enabled": True, "threshold_high": 4.0, "threshold_low": 0.5, "window": 2},
            {"enabled": False, "threshold_high": 4.0, "threshold_low": 0.5, "window": 2},
        ]
        tokens_a = list(range(10))
        tokens_b = list(range(10))
        meta = _sampling_meta([tokens_a, tokens_b])

        for _ in range(4):
            tokens_a.append(99)
            tokens_b.append(99)
            logits = _uniform_logits(batch=2)
            result = shim._apply_entropy_guardrail(logits, meta)

        assert result[0, EOS_ID] == 100.0
        assert result[1, EOS_ID] != 100.0


class TestStaleStatePruning:
    """Old request state is cleaned up when the batch shrinks."""

    def test_stale_entries_pruned(self):
        shim = _GuardrailShim(enabled=True, window=100)

        for i in range(20):
            fake_key = 1_000_000 + i
            shim._entropy_state[fake_key] = {
                "consecutive_oob": 0,
                "last_step": 0,
                "cfg": dict(shim._entropy_defaults),
            }

        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)
        shim._apply_entropy_guardrail(_uniform_logits(), meta)

        # 20 stale + 1 active = 21 > batch_size*2+10 = 12 -> pruning fires
        assert len(shim._entropy_state) <= 3, (
            f"Expected pruning, got {len(shim._entropy_state)} entries"
        )


class TestEdgeCases:
    def test_invalid_eos_id_is_noop(self):
        shim = _GuardrailShim(enabled=True, eos_id=-1)
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)
        logits = _uniform_logits()
        result = shim._apply_entropy_guardrail(logits, meta)
        assert torch.equal(result, logits)

    def test_eos_id_out_of_range_is_noop(self):
        shim = _GuardrailShim(enabled=True, eos_id=VOCAB_SIZE + 10)
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)
        logits = _uniform_logits()
        result = shim._apply_entropy_guardrail(logits, meta)
        assert torch.equal(result, logits)


class TestTriggerMetadata:
    """Verify that guardrail trigger metadata is stored correctly."""

    def test_trigger_stores_metadata(self):
        window = 2
        shim = _GuardrailShim(enabled=True, threshold_high=4.0, window=window)
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        # Run exactly `window` steps — trigger fires on the last one.
        for _ in range(window):
            tokens[0].append(99)
            shim._apply_entropy_guardrail(_uniform_logits(), meta)

        # Should have exactly one trigger for batch index 0
        assert 0 in shim._entropy_guardrail_triggered
        trigger = shim._entropy_guardrail_triggered[0]
        assert trigger["reason"] == "high_entropy"
        assert "step" in trigger
        assert "entropy" in trigger
        assert trigger["threshold"] == 4.0
        assert trigger["window"] == window

    def test_low_entropy_trigger_metadata(self):
        window = 2
        shim = _GuardrailShim(
            enabled=True, threshold_low=0.5, threshold_high=10.0, window=window
        )
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        for _ in range(window):
            tokens[0].append(99)
            shim._apply_entropy_guardrail(_peaked_logits(), meta)

        assert 0 in shim._entropy_guardrail_triggered
        trigger = shim._entropy_guardrail_triggered[0]
        assert trigger["reason"] == "low_entropy"
        assert trigger["threshold"] == 0.5

    def test_no_trigger_metadata_when_normal(self):
        shim = _GuardrailShim(enabled=True, threshold_high=6.0, threshold_low=0.1)
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        for _ in range(10):
            tokens[0].append(99)
            shim._apply_entropy_guardrail(_normal_logits(), meta)

        assert len(shim._entropy_guardrail_triggered) == 0

    def test_trigger_metadata_cleared_each_step(self):
        """Each call to _apply_entropy_guardrail resets the trigger dict."""
        window = 2
        shim = _GuardrailShim(enabled=True, threshold_high=4.0, window=window)
        tokens = [list(range(10))]
        meta = _sampling_meta(tokens)

        # Trigger
        for _ in range(window + 1):
            tokens[0].append(99)
            shim._apply_entropy_guardrail(_uniform_logits(), meta)

        # Next step should clear the trigger dict (even if not triggered again)
        tokens[0].append(99)
        shim._apply_entropy_guardrail(_normal_logits(), meta)
        assert len(shim._entropy_guardrail_triggered) == 0
