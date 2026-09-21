from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from patchpilot.agents.coder import CodingAgent
from patchpilot.llm.client import LLMResponse
from patchpilot.models import ActionType, CodeChange, Complexity, Plan, PlanStep, TokenUsage


@pytest.fixture
def coder(mock_llm_client: MagicMock) -> CodingAgent:
    return CodingAgent(llm_client=mock_llm_client)


@pytest.mark.asyncio
async def test_apply_plan_produces_changes(
    coder: CodingAgent, mock_llm_client: MagicMock, sample_plan: Plan
) -> None:
    mock_llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content='[{"file_path": "src/validators.py", "new_content": "def foo(): pass", "change_description": "added foo"}]',
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
    changes = await coder.apply_plan(sample_plan, repo_root="/tmp/repo")
    assert len(changes) >= 1
    assert isinstance(changes[0], CodeChange)


@pytest.mark.asyncio
async def test_apply_fix_returns_code_change(
    coder: CodingAgent, mock_llm_client: MagicMock
) -> None:
    mock_llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content='{"file_path": "src/foo.py", "new_content": "def fix(): pass", "change_description": "fixed"}',
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
    change = await coder.apply_fix("src/foo.py", "def bug(): pass", "fix the bug", "SyntaxError")
    assert isinstance(change, CodeChange)
    assert change.file_path == "src/foo.py"


def test_validate_python_syntax_valid(coder: CodingAgent) -> None:
    change = CodeChange(
        file_path="foo.py",
        original_content="",
        modified_content="def foo():\n    pass\n",
        change_description="valid python",
    )
    coder._validate_python_syntax(change)


def test_validate_python_syntax_invalid(coder: CodingAgent) -> None:
    change = CodeChange(
        file_path="foo.py",
        original_content="",
        modified_content="def foo()\n    pass\n",
        change_description="invalid python",
    )
    with pytest.raises(SyntaxError):
        coder._validate_python_syntax(change)


def test_topological_sort(coder: CodingAgent) -> None:
    step1 = PlanStep(
        step_id="1",
        description="first step",
        target_files=["a.py"],
        action_type=ActionType.MODIFY,
        rationale="r1",
        dependencies=[],
        estimated_complexity=Complexity.LOW,
    )
    step2 = PlanStep(
        step_id="2",
        description="second step",
        target_files=["b.py"],
        action_type=ActionType.MODIFY,
        rationale="r2",
        dependencies=["1"],
        estimated_complexity=Complexity.LOW,
    )
    sorted_steps = coder._topological_sort([step2, step1])
    assert sorted_steps[0].step_id == "1"
    assert sorted_steps[1].step_id == "2"


def test_get_latest_content_from_prior_changes(coder: CodingAgent) -> None:
    change = CodeChange(
        file_path="foo.py",
        original_content="",
        modified_content="updated content",
        change_description="modified",
    )
    content = coder._get_latest_content("foo.py", [change])
    assert content == "updated content"
