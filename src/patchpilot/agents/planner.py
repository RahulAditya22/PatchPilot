"""Planning Agent — breaks a GitHub issue into an actionable implementation plan.

This is the first agent in the PatchPilot pipeline. It takes a parsed issue
and codebase context, then produces a structured Plan with ordered steps.

Model Attribution: Claude Opus (core reasoning component)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from patchpilot.models import (
    ActionType,
    Complexity,
    IssueRequirement,
    Plan,
    PlanStep,
)

if TYPE_CHECKING:
    from patchpilot.llm.client import LLMClient
    from patchpilot.observability.tracer import AgentTracer

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an expert software engineering planning agent. Your job is to analyze
a GitHub issue and the relevant codebase context, then produce a detailed, actionable implementation plan.

## Your Process:
1. Understand the issue requirements thoroughly
2. Search the codebase to find relevant files and understand the architecture
3. Map dependencies between files/modules
4. Create a step-by-step plan with specific file changes

## Rules:
- Each step must target specific files
- Steps must be ordered by dependency (do prerequisites first)
- Estimate complexity for each step
- Explain your rationale for each decision
- If the issue is ambiguous, interpret it reasonably and note your assumptions
- Always consider test files that need to be updated or created

## Available Tools:
You can call these tools to gather information:
- search_code: Search the codebase semantically or lexically
- read_file: Read the contents of a specific file
- list_files: List files in a directory
- analyze_dependencies: Analyze imports/dependencies of a file

Respond with your plan as a structured JSON object."""


class PlanningAgent:
    """Agent that analyzes issues and produces implementation plans.

    Uses LLM tool-calling to search the codebase, understand the architecture,
    and produce a structured plan with ordered steps.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tracer: AgentTracer | None = None,
    ) -> None:
        """Initialize the planning agent.

        Args:
            llm_client: LLM client for generating plans.
            tracer: Optional tracer for observability.
        """
        self._llm = llm_client
        self._tracer = tracer
        self._tool_handlers: dict[str, Any] = {}

    def register_tool_handler(self, tool_name: str, handler: Any) -> None:
        """Register a handler function for a tool.

        Args:
            tool_name: Name of the tool.
            handler: Async callable that executes the tool.
        """
        self._tool_handlers[tool_name] = handler

    async def create_plan(
        self,
        issue: IssueRequirement,
        codebase_context: str,
        file_tree: list[str],
    ) -> Plan:
        """Create an implementation plan for the given issue.

        Args:
            issue: Parsed issue requirements.
            codebase_context: Summary of relevant codebase context from search.
            file_tree: List of all file paths in the repository.

        Returns:
            A structured Plan with ordered implementation steps.
        """
        step_id = None
        if self._tracer:
            step_id = self._tracer.start_step(
                "planning",
                input_summary=f"Issue: {issue.title}",
            )

        try:
            plan = await self._generate_plan(issue, codebase_context, file_tree)
            logger.info(
                "Created plan with %d steps for issue: %s",
                len(plan.steps),
                issue.title,
            )
            return plan
        finally:
            if self._tracer and step_id:
                self._tracer.end_step(
                    step_id,
                    output_summary=f"Plan with {len(plan.steps) if 'plan' in dir() else 0} steps",
                    model_used=self._llm.model,
                )

    async def _generate_plan(
        self,
        issue: IssueRequirement,
        codebase_context: str,
        file_tree: list[str],
    ) -> Plan:
        """Generate plan via LLM with tool-calling loop.

        Args:
            issue: The parsed issue.
            codebase_context: Context from codebase search.
            file_tree: Repository file tree.

        Returns:
            Structured Plan object.
        """
        from patchpilot.llm.schemas import PLANNER_TOOLS

        file_tree_str = "\n".join(file_tree[:500])  # Cap at 500 files
        user_message = self._build_user_message(issue, codebase_context, file_tree_str)

        messages: list[dict[str, str]] = [{"role": "user", "content": user_message}]
        max_tool_rounds = 10

        for round_num in range(max_tool_rounds):
            logger.debug("Planning round %d/%d", round_num + 1, max_tool_rounds)

            response = await self._llm.generate(
                messages=messages,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                tools=PLANNER_TOOLS,
                temperature=0.0,
            )

            # If no tool calls, the response should contain the plan
            if not response.tool_calls:
                return self._parse_plan_response(response.content, issue)

            # Process tool calls
            tool_results = await self._execute_tool_calls(response.tool_calls)

            # Add assistant response and tool results to conversation
            messages.append({"role": "assistant", "content": response.content})
            for tool_result in tool_results:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result for {tool_result['tool_name']}:\n{tool_result['result']}",
                    }
                )

        # If we exhausted tool rounds, try to parse whatever we have
        logger.warning("Exhausted %d tool-calling rounds, forcing plan generation", max_tool_rounds)
        final_response = await self._llm.generate(
            messages=messages
            + [
                {
                    "role": "user",
                    "content": "Please now provide your final plan as a JSON object. "
                    "No more tool calls needed.",
                }
            ],
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.0,
        )
        return self._parse_plan_response(final_response.content, issue)

    def _build_user_message(
        self,
        issue: IssueRequirement,
        codebase_context: str,
        file_tree_str: str,
    ) -> str:
        """Build the user message for the planning prompt.

        Args:
            issue: Parsed issue.
            codebase_context: Codebase context string.
            file_tree_str: File tree as string.

        Returns:
            Formatted user message.
        """
        return f"""## Issue to Implement

**Title:** {issue.title}
**Description:** {issue.description}
**Requirements:**
{chr(10).join(f"- {r}" for r in issue.requirements)}
**Priority:** {issue.priority}
**Labels:** {", ".join(issue.labels)}

## Repository File Tree
```
{file_tree_str}
```

## Codebase Context (from search)
{codebase_context}

## Instructions
Analyze this issue and create a detailed implementation plan. First use the available tools
to search the codebase and understand relevant code. Then produce your plan as a JSON object
with this structure:

```json
{{
    "issue_summary": "Brief summary of what needs to be done",
    "reasoning": "Your analysis and reasoning",
    "steps": [
        {{
            "step_id": "step_1",
            "description": "What to do in this step",
            "target_files": ["path/to/file.py"],
            "action_type": "MODIFY",
            "rationale": "Why this change is needed",
            "dependencies": [],
            "estimated_complexity": "LOW"
        }}
    ]
}}
```"""

    async def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Execute a batch of tool calls and return results.

        Args:
            tool_calls: List of tool call dicts with 'name' and 'arguments'.

        Returns:
            List of dicts with 'tool_name' and 'result' keys.
        """
        results: list[dict[str, str]] = []
        for call in tool_calls:
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            handler = self._tool_handlers.get(tool_name)
            if handler is None:
                result = f"Error: Unknown tool '{tool_name}'"
                logger.warning("Unknown tool called: %s", tool_name)
            else:
                try:
                    result = await handler(**arguments)
                    if not isinstance(result, str):
                        result = json.dumps(result, indent=2, default=str)
                except Exception as exc:
                    result = f"Error executing tool '{tool_name}': {exc}"
                    logger.error("Tool execution failed: %s - %s", tool_name, exc)

            results.append({"tool_name": tool_name, "result": result})

        return results

    def _parse_plan_response(self, content: str, issue: IssueRequirement) -> Plan:
        """Parse the LLM response into a Plan object.

        Args:
            content: Raw LLM response text.
            issue: Original issue for context.

        Returns:
            Parsed Plan object.

        Raises:
            ValueError: If the response cannot be parsed into a valid plan.
        """
        # Try to extract JSON from the response
        plan_json = self._extract_json(content)

        if plan_json is None:
            logger.warning("Could not extract JSON from planning response, creating minimal plan")
            return Plan(
                issue_summary=issue.title,
                steps=[
                    PlanStep(
                        step_id="step_1",
                        description=f"Implement: {issue.title}",
                        target_files=[],
                        action_type=ActionType.MODIFY,
                        rationale="Auto-generated fallback plan from unstructured response",
                        dependencies=[],
                        estimated_complexity=Complexity.MEDIUM,
                    )
                ],
                reasoning=content[:1000],
            )

        try:
            steps = []
            for i, step_data in enumerate(plan_json.get("steps", [])):
                action_str = step_data.get("action_type", "MODIFY").upper()
                try:
                    action_type = ActionType[action_str]
                except KeyError:
                    action_type = ActionType.MODIFY

                complexity_str = step_data.get("estimated_complexity", "MEDIUM").upper()
                try:
                    complexity = Complexity[complexity_str]
                except KeyError:
                    complexity = Complexity.MEDIUM

                steps.append(
                    PlanStep(
                        step_id=step_data.get("step_id", f"step_{i + 1}"),
                        description=step_data.get("description", ""),
                        target_files=step_data.get("target_files", []),
                        action_type=action_type,
                        rationale=step_data.get("rationale", ""),
                        dependencies=step_data.get("dependencies", []),
                        estimated_complexity=complexity,
                    )
                )

            return Plan(
                issue_summary=plan_json.get("issue_summary", issue.title),
                steps=steps,
                reasoning=plan_json.get("reasoning", ""),
            )
        except Exception as exc:
            raise ValueError(f"Failed to parse plan JSON: {exc}") from exc

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from text that may contain markdown code blocks.

        Args:
            text: Raw text possibly containing JSON.

        Returns:
            Parsed dict or None if no valid JSON found.
        """
        # Try direct parse first
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        import re

        json_patterns = [
            r"```json\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
            r"\{[\s\S]*\}",
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    continue

        return None
