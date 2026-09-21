from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from patchpilot.agents.orchestrator import Orchestrator
from patchpilot.models import (
    AgentResult,
    Diagnosis,
    IssueRequirement,
    Plan,
    RepoInfo,
    RootCause,
    SecurityReport,
    TestResult,
)


@pytest.fixture
def orchestrator() -> Orchestrator:
    orch = Orchestrator()
    orch._components_initialized = True

    orch._planner = MagicMock()
    orch._planner.create_plan = AsyncMock()
    orch._planner.register_tool_handler = MagicMock()

    orch._coder = MagicMock()
    orch._coder.apply_plan = AsyncMock()
    orch._coder.apply_fix = AsyncMock()
    orch._coder.set_file_reader = MagicMock()

    orch._analyst = MagicMock()
    orch._analyst.analyze = AsyncMock()

    orch._repo_ingester = MagicMock()
    orch._repo_ingester.ingest = AsyncMock(
        return_value=RepoInfo(
            clone_path="/tmp/fake_repo",
            repo_url="https://github.com/test/test",
            default_branch="main",
            file_tree=["src/main.py"],
            language_stats={"Python": 100},
        )
    )

    orch._indexer = MagicMock()
    orch._indexer.index_repository = AsyncMock(return_value=1)

    orch._code_search = MagicMock()
    orch._code_search.search = AsyncMock(return_value=[])

    orch._issue_parser = MagicMock()
    orch._issue_parser.parse = AsyncMock()

    orch._sandbox = MagicMock()
    orch._sandbox.run_tests = AsyncMock()

    orch._diff_generator = MagicMock()
    orch._diff_generator.generate_unified_diff = MagicMock(return_value="diff --git a/b")

    orch._security_checker = MagicMock()
    orch._security_checker.check = AsyncMock(return_value=SecurityReport(passed=True, findings=[]))

    orch._dep_analyzer = MagicMock()

    orch._tracer = MagicMock()
    orch._tracer.start_step = MagicMock(return_value="step_1")
    orch._tracer.end_step = MagicMock()
    orch._tracer.get_traces = MagicMock(return_value=[])

    orch._token_tracker = MagicMock()
    orch._token_tracker.get_all_usage = MagicMock(return_value=[])

    return orch


@pytest.mark.asyncio
async def test_run_success_flow(
    orchestrator: Orchestrator, sample_issue: IssueRequirement, sample_plan: Plan
) -> None:
    orchestrator._issue_parser.parse = AsyncMock(return_value=sample_issue)
    orchestrator._planner.create_plan = AsyncMock(return_value=sample_plan)
    orchestrator._coder.apply_plan = AsyncMock(return_value=[])
    orchestrator._sandbox.run_tests = AsyncMock(
        return_value=TestResult(
            passed=True,
            exit_code=0,
            stdout="OK",
            stderr="",
            duration_seconds=0.1,
            num_passed=1,
            num_failed=0,
            num_errors=0,
            failing_tests=[],
        )
    )

    result = await orchestrator.run("https://github.com/test/test", "Fix the issue", max_retries=1)
    assert isinstance(result, AgentResult)
    assert result.success is True


@pytest.mark.asyncio
async def test_run_with_test_failure_and_retry(
    orchestrator: Orchestrator, sample_issue: IssueRequirement, sample_plan: Plan
) -> None:
    orchestrator._issue_parser.parse = AsyncMock(return_value=sample_issue)
    orchestrator._planner.create_plan = AsyncMock(return_value=sample_plan)
    orchestrator._coder.apply_plan = AsyncMock(return_value=[])
    orchestrator._sandbox.run_tests = AsyncMock(
        side_effect=[
            TestResult(
                passed=False,
                exit_code=1,
                stdout="",
                stderr="err",
                duration_seconds=0.1,
                num_passed=0,
                num_failed=1,
                num_errors=0,
                failing_tests=[],
            ),
            TestResult(
                passed=True,
                exit_code=0,
                stdout="OK",
                stderr="",
                duration_seconds=0.1,
                num_passed=1,
                num_failed=0,
                num_errors=0,
                failing_tests=[],
            ),
        ]
    )
    orchestrator._analyst.analyze = AsyncMock(
        return_value=Diagnosis(
            root_cause=RootCause.LOGIC_ERROR,
            explanation="logic error",
            suggested_fixes=[],
            confidence=0.9,
        )
    )

    result = await orchestrator.run("https://github.com/test/test", "Fix issue", max_retries=1)
    assert result.success is True
    assert orchestrator._analyst.analyze.called


@pytest.mark.asyncio
async def test_run_max_retries_exhausted(
    orchestrator: Orchestrator, sample_issue: IssueRequirement, sample_plan: Plan
) -> None:
    orchestrator._issue_parser.parse = AsyncMock(return_value=sample_issue)
    orchestrator._planner.create_plan = AsyncMock(return_value=sample_plan)
    orchestrator._coder.apply_plan = AsyncMock(return_value=[])
    orchestrator._sandbox.run_tests = AsyncMock(
        return_value=TestResult(
            passed=False,
            exit_code=1,
            stdout="",
            stderr="err",
            duration_seconds=0.1,
            num_passed=0,
            num_failed=1,
            num_errors=0,
            failing_tests=[],
        )
    )
    orchestrator._analyst.analyze = AsyncMock(
        return_value=Diagnosis(
            root_cause=RootCause.LOGIC_ERROR,
            explanation="logic error",
            suggested_fixes=[],
            confidence=0.9,
        )
    )

    result = await orchestrator.run("https://github.com/test/test", "Fix issue", max_retries=1)
    assert result.success is False


@pytest.mark.asyncio
async def test_pipeline_error_returns_failed_result(orchestrator: Orchestrator) -> None:
    orchestrator._repo_ingester.ingest = AsyncMock(side_effect=Exception("Cloning failed"))
    result = await orchestrator.run("https://github.com/test/test", "Fix issue")
    assert result.success is False
    assert "Cloning failed" in (result.error_message or "")


def test_write_changes_creates_files(tmp_path: os.PathLike) -> None:
    from patchpilot.models import CodeChange

    change = CodeChange(
        file_path="sub/test.py",
        original_content="",
        modified_content="print('hello')",
        change_description="created test.py",
    )
    Orchestrator._write_changes([change], str(tmp_path))

    target = tmp_path / "sub" / "test.py"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "print('hello')"
