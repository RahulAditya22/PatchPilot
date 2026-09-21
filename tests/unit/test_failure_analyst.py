from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from patchpilot.agents.failure_analyst import FailureAnalyst
from patchpilot.llm.client import LLMResponse
from patchpilot.models import RootCause, TestResult, TokenUsage


@pytest.fixture
def analyst(mock_llm_client: MagicMock) -> FailureAnalyst:
    return FailureAnalyst(llm_client=mock_llm_client)


def test_heuristic_syntax_error(analyst: FailureAnalyst) -> None:
    result = TestResult(
        passed=False,
        exit_code=1,
        stdout="",
        stderr="SyntaxError: invalid syntax",
        duration_seconds=0.1,
        num_passed=0,
        num_failed=1,
        num_errors=0,
        failing_tests=[],
    )
    diagnosis = analyst._heuristic_analysis(result)
    assert diagnosis.root_cause == RootCause.SYNTAX_ERROR


def test_heuristic_import_error(analyst: FailureAnalyst) -> None:
    result = TestResult(
        passed=False,
        exit_code=1,
        stdout="",
        stderr="ImportError: cannot import name 'foo'",
        duration_seconds=0.1,
        num_passed=0,
        num_failed=1,
        num_errors=0,
        failing_tests=[],
    )
    diagnosis = analyst._heuristic_analysis(result)
    assert diagnosis.root_cause == RootCause.IMPORT_ERROR


def test_heuristic_assertion_error(analyst: FailureAnalyst) -> None:
    result = TestResult(
        passed=False,
        exit_code=1,
        stdout="AssertionError: assert False == True",
        stderr="",
        duration_seconds=0.1,
        num_passed=0,
        num_failed=1,
        num_errors=0,
        failing_tests=[],
    )
    diagnosis = analyst._heuristic_analysis(result)
    assert diagnosis.root_cause == RootCause.TEST_ASSERTION


def test_heuristic_timeout(analyst: FailureAnalyst) -> None:
    result = TestResult(
        passed=False,
        exit_code=1,
        stdout="",
        stderr="TimeoutError: command timed out",
        duration_seconds=30.0,
        num_passed=0,
        num_failed=1,
        num_errors=0,
        failing_tests=[],
    )
    diagnosis = analyst._heuristic_analysis(result)
    assert diagnosis.root_cause == RootCause.TIMEOUT


def test_heuristic_runtime_crash(analyst: FailureAnalyst) -> None:
    result = TestResult(
        passed=False,
        exit_code=1,
        stdout="",
        stderr="TypeError: unsupported operand type",
        duration_seconds=0.1,
        num_passed=0,
        num_failed=1,
        num_errors=0,
        failing_tests=[],
    )
    diagnosis = analyst._heuristic_analysis(result)
    assert diagnosis.root_cause == RootCause.RUNTIME_CRASH


def test_heuristic_unknown(analyst: FailureAnalyst) -> None:
    result = TestResult(
        passed=False,
        exit_code=1,
        stdout="",
        stderr="Something unusual happened",
        duration_seconds=0.1,
        num_passed=0,
        num_failed=1,
        num_errors=0,
        failing_tests=[],
    )
    diagnosis = analyst._heuristic_analysis(result)
    assert diagnosis.root_cause == RootCause.UNKNOWN


@pytest.mark.asyncio
async def test_high_confidence_skips_llm(
    analyst: FailureAnalyst, mock_llm_client: MagicMock
) -> None:
    result = TestResult(
        passed=False,
        exit_code=1,
        stdout="",
        stderr="SyntaxError: invalid syntax",
        duration_seconds=0.1,
        num_passed=0,
        num_failed=1,
        num_errors=0,
        failing_tests=[],
    )
    diagnosis = await analyst.analyze(result, changes=[])
    assert diagnosis.root_cause == RootCause.SYNTAX_ERROR
    assert not mock_llm_client.generate.called


@pytest.mark.asyncio
async def test_llm_analysis_called_for_low_confidence(
    analyst: FailureAnalyst, mock_llm_client: MagicMock
) -> None:
    mock_llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content="""{
                "root_cause": "LOGIC_ERROR",
                "explanation": "Off by one error in loop",
                "confidence": 0.85,
                "suggested_fixes": []
            }""",
            tool_calls=None,
            token_usage=TokenUsage(
                model="claude-sonnet-4-20250514",
                input_tokens=10,
                output_tokens=10,
                estimated_cost_usd=0.001,
            ),
            raw_response=None,
        )
    )
    result = TestResult(
        passed=False,
        exit_code=1,
        stdout="",
        stderr="Something unusual happened",
        duration_seconds=0.1,
        num_passed=0,
        num_failed=1,
        num_errors=0,
        failing_tests=[],
    )
    diagnosis = await analyst.analyze(result, changes=[])
    assert mock_llm_client.generate.called
    assert diagnosis.root_cause == RootCause.LOGIC_ERROR
