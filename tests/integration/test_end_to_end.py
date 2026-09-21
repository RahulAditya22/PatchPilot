from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from patchpilot.agents.orchestrator import Orchestrator
from patchpilot.models import (
    ActionType,
    Complexity,
    IssueRequirement,
    Plan,
    PlanStep,
    RepoInfo,
    SecurityReport,
    TestResult,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_simple_bugfix(temp_repo: pytest.TempPathFactory) -> None:
    """Mock LLM and external tools, run orchestrator end-to-end, verify AgentResult."""
    orch = Orchestrator()
    await orch._init_components()

    orch._repo_ingester.ingest = AsyncMock(
        return_value=RepoInfo(
            clone_path=str(temp_repo),
            repo_url=str(temp_repo),
            default_branch="main",
            file_tree=["src/main.py"],
            language_stats={"Python": 100},
        )
    )
    orch._indexer.index_repository = AsyncMock(return_value=1)
    orch._code_search.search = AsyncMock(return_value=[])

    orch._issue_parser.parse = AsyncMock(
        return_value=IssueRequirement(
            title="Fix bug in main",
            description="Fix return value of hello()",
            requirements=["Fix hello()"],
            labels=[],
            priority="medium",
        )
    )

    orch._planner.create_plan = AsyncMock(
        return_value=Plan(
            issue_summary="Fix hello()",
            steps=[
                PlanStep(
                    step_id="step_1",
                    description="Fix hello return value",
                    target_files=["src/main.py"],
                    action_type=ActionType.MODIFY,
                    rationale="Return correct string",
                    dependencies=[],
                    estimated_complexity=Complexity.LOW,
                )
            ],
            reasoning="Simple fix",
        )
    )

    from patchpilot.models import CodeChange

    orch._coder.apply_plan = AsyncMock(
        return_value=[
            CodeChange(
                file_path="src/main.py",
                original_content="def hello(): return 'world'",
                modified_content="def hello(): return 'Hello, World!'",
                change_description="Fixed hello",
            )
        ]
    )

    orch._sandbox.run_tests = AsyncMock(
        return_value=TestResult(
            passed=True,
            exit_code=0,
            stdout="1 passed",
            stderr="",
            duration_seconds=0.2,
            num_passed=1,
            num_failed=0,
            num_errors=0,
            failing_tests=[],
        )
    )

    orch._security_checker.check = AsyncMock(return_value=SecurityReport(passed=True, findings=[]))

    result = await orch.run(str(temp_repo), "Fix bug in main")
    assert result.success is True
    assert result.diff_text != ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_new_feature(temp_repo: pytest.TempPathFactory) -> None:
    """Mock LLM, run with a feature request, verify changes are created."""
    orch = Orchestrator()
    await orch._init_components()

    orch._repo_ingester.ingest = AsyncMock(
        return_value=RepoInfo(
            clone_path=str(temp_repo),
            repo_url=str(temp_repo),
            default_branch="main",
            file_tree=["src/main.py"],
            language_stats={"Python": 100},
        )
    )
    orch._indexer.index_repository = AsyncMock(return_value=1)
    orch._code_search.search = AsyncMock(return_value=[])

    orch._issue_parser.parse = AsyncMock(
        return_value=IssueRequirement(
            title="Add feature",
            description="Add goodbye function",
            requirements=["Add goodbye()"],
            labels=[],
            priority="medium",
        )
    )

    orch._planner.create_plan = AsyncMock(
        return_value=Plan(
            issue_summary="Add goodbye()",
            steps=[
                PlanStep(
                    step_id="step_1",
                    description="Add goodbye function",
                    target_files=["src/main.py"],
                    action_type=ActionType.MODIFY,
                    rationale="Feature request",
                    dependencies=[],
                    estimated_complexity=Complexity.LOW,
                )
            ],
            reasoning="Add feature",
        )
    )

    from patchpilot.models import CodeChange

    orch._coder.apply_plan = AsyncMock(
        return_value=[
            CodeChange(
                file_path="src/main.py",
                original_content="def hello(): return 'hello'",
                modified_content="def hello(): return 'hello'\ndef goodbye(): return 'goodbye'",
                change_description="Added goodbye function",
            )
        ]
    )

    orch._sandbox.run_tests = AsyncMock(
        return_value=TestResult(
            passed=True,
            exit_code=0,
            stdout="2 passed",
            stderr="",
            duration_seconds=0.2,
            num_passed=2,
            num_failed=0,
            num_errors=0,
            failing_tests=[],
        )
    )

    orch._security_checker.check = AsyncMock(return_value=SecurityReport(passed=True, findings=[]))

    result = await orch.run(str(temp_repo), "Add goodbye feature")
    assert result.success is True
    assert len(result.changes) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_graceful_failure(temp_repo: pytest.TempPathFactory) -> None:
    """Mock LLM error or ingester error, verify graceful failure."""
    orch = Orchestrator()
    await orch._init_components()

    orch._repo_ingester.ingest = AsyncMock(side_effect=RuntimeError("Clone failed"))

    result = await orch.run(str(temp_repo), "cause error")
    assert result.success is False
    assert result.error_message is not None
