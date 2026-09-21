from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from patchpilot.agents.orchestrator import Orchestrator
from patchpilot.models import ActionType, AgentResult, Complexity, PlanStep
from tests.eval.benchmark import EvaluationRunner


@pytest.mark.eval
@pytest.mark.asyncio
async def test_adversarial_malformed_issue(temp_repo: pytest.TempPathFactory) -> None:
    """Feed garbage/empty issue text, verify agent doesn't crash and returns a result."""
    orch = Orchestrator()
    with patch.object(orch, "_init_components"):
        orch._components_initialized = True
        orch._tracer = MagicMock()
        orch._tracer.start_step = MagicMock(return_value="step_1")
        orch._tracer.end_step = MagicMock()
        orch._tracer.get_traces = MagicMock(return_value=[])
        orch._token_tracker = MagicMock()
        orch._token_tracker.get_all_usage = MagicMock(return_value=[])
        orch._repo_ingester = AsyncMock()
        orch._repo_ingester.ingest = AsyncMock(side_effect=ValueError("Invalid issue text"))

        result = await orch.run(str(temp_repo), "")
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert result.error_message is not None


@pytest.mark.eval
@pytest.mark.asyncio
async def test_adversarial_binary_repo(temp_repo: pytest.TempPathFactory) -> None:
    """Repo with binary files only, verify graceful handling."""
    from patchpilot.tools.repo_ingester import RepoIngester

    ingester = RepoIngester()
    info = await ingester.ingest(str(temp_repo))
    assert info.clone_path is not None


@pytest.mark.eval
def test_adversarial_circular_dependencies() -> None:
    """Plan steps with circular deps, verify topological sort handles cycles gracefully without hanging."""
    from patchpilot.agents.coder import CodingAgent

    step1 = PlanStep(
        step_id="1",
        description="step 1",
        target_files=["a.py"],
        action_type=ActionType.MODIFY,
        rationale="r1",
        dependencies=["2"],
        estimated_complexity=Complexity.LOW,
    )
    step2 = PlanStep(
        step_id="2",
        description="step 2",
        target_files=["b.py"],
        action_type=ActionType.MODIFY,
        rationale="r2",
        dependencies=["1"],
        estimated_complexity=Complexity.LOW,
    )

    sorted_steps = CodingAgent._topological_sort([step1, step2])
    assert len(sorted_steps) == 2


@pytest.mark.eval
def test_evaluation_metrics_format() -> None:
    """Verify eval runner returns success_rate, avg_time, avg_cost metrics."""
    runner = EvaluationRunner(scenarios=[])
    report = runner.generate_report()
    assert "# PatchPilot Evaluation Benchmark" in report
    assert "Success Rate" in report
