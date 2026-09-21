from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from patchpilot.agents.planner import PlanningAgent
from patchpilot.llm.client import LLMResponse
from patchpilot.models import IssueRequirement, Plan, TokenUsage


@pytest.fixture
def planner(mock_llm_client: MagicMock) -> PlanningAgent:
    return PlanningAgent(llm_client=mock_llm_client)


@pytest.mark.asyncio
async def test_create_plan_returns_plan(
    planner: PlanningAgent, mock_llm_client: MagicMock, sample_issue: IssueRequirement
) -> None:
    mock_llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content="""{
                "issue_summary": "Add input validation",
                "reasoning": "Need to validate input",
                "steps": [
                    {
                        "step_id": "step_1",
                        "description": "Add email validation",
                        "target_files": ["src/validators.py"],
                        "action_type": "MODIFY",
                        "rationale": "Validation needed",
                        "dependencies": [],
                        "estimated_complexity": "LOW"
                    }
                ]
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
    plan = await planner.create_plan(
        sample_issue, codebase_context="some context", file_tree=["src/validators.py"]
    )
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "step_1"


@pytest.mark.asyncio
async def test_create_plan_with_tool_calls(
    planner: PlanningAgent, mock_llm_client: MagicMock, sample_issue: IssueRequirement
) -> None:
    async def fake_search(query: str, search_type: str = "hybrid", max_results: int = 10) -> str:
        return "search results"

    planner.register_tool_handler("search_code", fake_search)

    mock_llm_client.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Searching code...",
                tool_calls=[{"name": "search_code", "arguments": {"query": "foo"}}],
                token_usage=TokenUsage(
                    model="claude-sonnet-4-20250514",
                    input_tokens=10,
                    output_tokens=10,
                    estimated_cost_usd=0.001,
                ),
                raw_response=None,
            ),
            LLMResponse(
                content="""{
                    "issue_summary": "Add validation",
                    "reasoning": "Done research",
                    "steps": []
                }""",
                tool_calls=None,
                token_usage=TokenUsage(
                    model="claude-sonnet-4-20250514",
                    input_tokens=10,
                    output_tokens=10,
                    estimated_cost_usd=0.001,
                ),
                raw_response=None,
            ),
        ]
    )
    plan = await planner.create_plan(sample_issue, codebase_context="", file_tree=[])
    assert isinstance(plan, Plan)


def test_parse_plan_from_code_block(planner: PlanningAgent) -> None:
    content = """```json
    {
        "issue_summary": "test",
        "reasoning": "test",
        "steps": []
    }
    ```"""
    extracted = planner._extract_json(content)
    assert extracted is not None
    assert "issue_summary" in extracted


def test_parse_plan_fallback(planner: PlanningAgent, sample_issue: IssueRequirement) -> None:
    content = "invalid json text"
    plan = planner._parse_plan_response(content, sample_issue)
    assert isinstance(plan, Plan)
    assert len(plan.steps) >= 1


def test_build_user_message_contains_issue_info(
    planner: PlanningAgent, sample_issue: IssueRequirement
) -> None:
    msg = planner._build_user_message(sample_issue, codebase_context="ctx", file_tree_str="tree")
    assert sample_issue.title in msg
    assert sample_issue.description in msg
