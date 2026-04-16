"""Shared test fixtures and helpers."""

import pathlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from gandalf.models import (
    BatchJudgeInput,
    CriterionResult,
    GraderConfig,
    LLMUsage,
)


@pytest.fixture(autouse=True)
def _stub_preflight() -> Iterator[None]:
    """Silence the orchestrator preflight check during tests by default.

    Tests that exercise the preflight or the underlying probes import them by
    name and therefore bypass this patch of the orchestrator attribute.
    """
    with patch("gandalf.orchestrator.preflight_check"):
        yield


def make_config(**overrides: Any) -> GraderConfig:
    """Create a GraderConfig with sensible defaults for testing."""
    defaults: dict[str, Any] = {
        "instructions": "test",
        "rubric_path": "/rubric.json",
        "workdir": "/workspace",
        "trajectory_path": "/logs/trajectory.json",
        "sandbox_user": "sandbox",
        "output_dir": "/logs/grader",
    }
    defaults.update(overrides)
    return GraderConfig(**defaults)


def cr(*, weight: float, met: bool | None) -> CriterionResult:
    """Helper to build a CriterionResult for scoring tests."""
    return CriterionResult(
        criterion="test",
        weight=weight,
        met=met,
        reasoning="test",
    )


def make_batch_input(tmp_path: pathlib.Path, n: int = 2) -> BatchJudgeInput:
    """Create a BatchJudgeInput with *n* criteria rooted in tmp_path."""
    return BatchJudgeInput(
        model="test-model",
        instructions="do a thing",
        final_output="done",
        criteria=[f"criterion {i}" for i in range(n)],
        workdir=str(tmp_path),
    )


MOCK_USAGE = LLMUsage(
    cost_usd=0.05,
    prompt_tokens=1000,
    completion_tokens=500,
    cache_read_tokens=200,
)
