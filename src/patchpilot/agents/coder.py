"""Coding Agent — applies planned code changes to repository files.

Takes a Plan from the Planning Agent and the actual file contents,
then generates concrete code modifications using LLM-guided editing.

Model Attribution: Claude Opus (core code-modification engine)
"""

from __future__ import annotations

import ast
import json
import logging
from typing import TYPE_CHECKING, Any

from patchpilot.models import ActionType, CodeChange, Plan, PlanStep

if TYPE_CHECKING:
    from patchpilot.llm.client import LLMClient
    from patchpilot.observability.tracer import AgentTracer

logger = logging.getLogger(__name__)

CODER_SYSTEM_PROMPT = """You are an expert software engineer. Your job is to implement specific code changes
based on a plan step. You will receive:
1. The plan step describing what to change
2. The current contents of the target file(s)
3. Context about the broader plan

## Rules:
- Make minimal, focused changes that accomplish the step
- Preserve existing code style and conventions
- Add or update docstrings as needed
- Add type hints to any new code
- Ensure imports are correct and complete
- Do NOT remove or modify unrelated code
- If creating a new file, include proper module docstring and imports
- If creating tests, follow pytest conventions

## Output Format:
For each file you modify, respond with a JSON array of changes:
```json
[
    {
        "file_path": "path/to/file.py",
        "new_content": "complete new file content...",
        "change_description": "what was changed and why"
    }
]
```

Return the COMPLETE file content for each modified file (not just the diff).
Ensure the code is syntactically valid."""


class CodingAgent:
    """Agent that generates concrete code modifications from a plan.

    Uses LLM to produce file-level changes, with syntax validation
    for Python files.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tracer: AgentTracer | None = None,
    ) -> None:
        """Initialize the coding agent.

        Args:
            llm_client: LLM client for generating code.
            tracer: Optional tracer for observability.
        """
        self._llm = llm_client
        self._tracer = tracer
        self._file_reader: Any = None

    def set_file_reader(self, reader: Any) -> None:
        """Set the file reader callable for reading repo files.

        Args:
            reader: Async callable(file_path: str) -> str.
        """
        self._file_reader = reader

    async def apply_plan(
        self,
        plan: Plan,
        repo_root: str,
    ) -> list[CodeChange]:
        """Apply all steps in a plan, producing code changes.

        Args:
            plan: The implementation plan from the PlanningAgent.
            repo_root: Root directory of the repository.

        Returns:
            List of all code changes across all plan steps.
        """
        all_changes: list[CodeChange] = []

        # Sort steps by dependency order
        ordered_steps = self._topological_sort(plan.steps)

        for step in ordered_steps:
            step_id = None
            if self._tracer:
                step_id = self._tracer.start_step(
                    f"coding_step_{step.step_id}",
                    input_summary=step.description,
                )

            try:
                changes = await self._apply_step(step, plan, repo_root, all_changes)
                all_changes.extend(changes)
                logger.info(
                    "Step %s produced %d changes: %s",
                    step.step_id,
                    len(changes),
                    step.description,
                )
            except Exception as exc:
                logger.error("Failed to apply step %s: %s", step.step_id, exc)
                raise
            finally:
                if self._tracer and step_id:
                    self._tracer.end_step(
                        step_id,
                        output_summary=f"{len(changes) if 'changes' in dir() else 0} file changes",
                        model_used=self._llm.model,
                    )

        return all_changes

    async def apply_fix(
        self,
        file_path: str,
        current_content: str,
        fix_description: str,
        error_context: str,
    ) -> CodeChange:
        """Apply a targeted fix to a single file based on error analysis.

        Used by the retry loop when the FailureAnalyst identifies a fix.

        Args:
            file_path: Path to the file to fix.
            current_content: Current content of the file.
            fix_description: Description of what needs to be fixed.
            error_context: Error output that prompted the fix.

        Returns:
            A CodeChange with the fix applied.
        """
        step_id = None
        if self._tracer:
            step_id = self._tracer.start_step(
                "apply_fix",
                input_summary=f"Fix {file_path}: {fix_description[:100]}",
            )

        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""Fix the following file based on the error analysis.

## File: {file_path}
```
{current_content}
```

## Error Context:
```
{error_context}
```

## Required Fix:
{fix_description}

## Instructions:
Apply the fix and return the complete corrected file content as JSON:
```json
{{
    "file_path": "{file_path}",
    "new_content": "complete corrected file content...",
    "change_description": "description of fix applied"
}}
```""",
                }
            ]

            response = await self._llm.generate(
                messages=messages,
                system_prompt=CODER_SYSTEM_PROMPT,
                temperature=0.0,
            )

            change = self._parse_single_change(response.content, file_path, current_content)
            return change
        finally:
            if self._tracer and step_id:
                self._tracer.end_step(
                    step_id,
                    output_summary=f"Fixed {file_path}",
                    model_used=self._llm.model,
                )

    async def _apply_step(
        self,
        step: PlanStep,
        plan: Plan,
        repo_root: str,
        prior_changes: list[CodeChange],
    ) -> list[CodeChange]:
        """Apply a single plan step.

        Args:
            step: The plan step to apply.
            plan: The full plan for context.
            repo_root: Repository root path.
            prior_changes: Changes from prior steps (for context).

        Returns:
            List of code changes for this step.
        """
        # Gather current file contents
        file_contents: dict[str, str] = {}
        for target_file in step.target_files:
            # Check if a prior change already modified this file
            prior_content = self._get_latest_content(target_file, prior_changes)
            if prior_content is not None:
                file_contents[target_file] = prior_content
            elif step.action_type != ActionType.CREATE and self._file_reader:
                try:
                    content = await self._file_reader(target_file)
                    file_contents[target_file] = content
                except Exception:
                    file_contents[target_file] = ""
                    logger.warning("Could not read file: %s", target_file)

        # Build context about prior changes
        prior_context = ""
        if prior_changes:
            summaries = [f"- {c.file_path}: {c.change_description}" for c in prior_changes[-5:]]
            prior_context = "\n## Prior Changes Applied:\n" + "\n".join(summaries)

        # Build the prompt
        files_section = ""
        for path, content in file_contents.items():
            files_section += f"\n### File: {path}\n```\n{content}\n```\n"

        if step.action_type == ActionType.CREATE:
            files_section += f"\n### File to create: {', '.join(step.target_files)}\n(New file)\n"

        messages = [
            {
                "role": "user",
                "content": f"""## Plan Context
**Issue:** {plan.issue_summary}
**Overall reasoning:** {plan.reasoning[:500]}

## Current Step
**Step ID:** {step.step_id}
**Description:** {step.description}
**Action:** {step.action_type.value}
**Rationale:** {step.rationale}

## Target Files
{files_section}
{prior_context}

## Instructions
Implement this step. Return a JSON array of all file changes:
```json
[
    {{
        "file_path": "path/to/file.py",
        "new_content": "complete file content with changes...",
        "change_description": "what was changed"
    }}
]
```

Return the COMPLETE new content for each file.""",
            }
        ]

        response = await self._llm.generate(
            messages=messages,
            system_prompt=CODER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=8192,
        )

        changes = self._parse_changes_response(response.content, file_contents)

        # Validate Python syntax for .py files
        for change in changes:
            if change.file_path.endswith(".py"):
                self._validate_python_syntax(change)

        return changes

    def _parse_changes_response(
        self,
        content: str,
        original_contents: dict[str, str],
    ) -> list[CodeChange]:
        """Parse LLM response into CodeChange objects.

        Args:
            content: Raw LLM response.
            original_contents: Map of file paths to original contents.

        Returns:
            List of parsed code changes.

        Raises:
            ValueError: If response cannot be parsed.
        """
        json_data = self._extract_json_array(content)
        if json_data is None:
            raise ValueError("Could not parse code changes from LLM response")

        changes: list[CodeChange] = []
        for item in json_data:
            file_path = item.get("file_path", "")
            new_content = item.get("new_content", "")
            description = item.get("change_description", "")
            original = original_contents.get(file_path, "")

            changes.append(
                CodeChange(
                    file_path=file_path,
                    original_content=original,
                    modified_content=new_content,
                    change_description=description,
                )
            )

        return changes

    def _parse_single_change(
        self,
        content: str,
        file_path: str,
        original_content: str,
    ) -> CodeChange:
        """Parse a single change from LLM response.

        Args:
            content: Raw LLM response.
            file_path: Expected file path.
            original_content: Original file content.

        Returns:
            Parsed CodeChange.
        """
        # Try JSON parse
        json_data = self._extract_json_object(content)
        if json_data and "new_content" in json_data:
            return CodeChange(
                file_path=json_data.get("file_path", file_path),
                original_content=original_content,
                modified_content=json_data["new_content"],
                change_description=json_data.get("change_description", "Applied fix"),
            )

        # Try array parse
        json_array = self._extract_json_array(content)
        if json_array and len(json_array) > 0:
            item = json_array[0]
            return CodeChange(
                file_path=item.get("file_path", file_path),
                original_content=original_content,
                modified_content=item.get("new_content", original_content),
                change_description=item.get("change_description", "Applied fix"),
            )

        # Fallback: treat entire response as the new content
        logger.warning("Could not parse structured response, using raw content as fix")
        return CodeChange(
            file_path=file_path,
            original_content=original_content,
            modified_content=content,
            change_description="Applied fix (raw response)",
        )

    @staticmethod
    def _validate_python_syntax(change: CodeChange) -> None:
        """Validate Python syntax of a code change.

        Args:
            change: The code change to validate.

        Raises:
            SyntaxError: If the modified content has invalid Python syntax.
        """
        try:
            ast.parse(change.modified_content)
        except SyntaxError as exc:
            logger.error(
                "Syntax error in %s at line %d: %s",
                change.file_path,
                exc.lineno or 0,
                exc.msg,
            )
            raise

    @staticmethod
    def _get_latest_content(file_path: str, prior_changes: list[CodeChange]) -> str | None:
        """Get the most recent content for a file from prior changes.

        Args:
            file_path: The file to look up.
            prior_changes: List of prior changes.

        Returns:
            Latest content or None if not modified.
        """
        for change in reversed(prior_changes):
            if change.file_path == file_path:
                return change.modified_content
        return None

    @staticmethod
    def _topological_sort(steps: list[PlanStep]) -> list[PlanStep]:
        """Sort plan steps in dependency order.

        Args:
            steps: Unordered plan steps.

        Returns:
            Steps sorted so dependencies come first.
        """
        step_map = {s.step_id: s for s in steps}
        visited: set[str] = set()
        result: list[PlanStep] = []

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            visited.add(step_id)
            step = step_map.get(step_id)
            if step is None:
                return
            for dep_id in step.dependencies:
                visit(dep_id)
            result.append(step)

        for step in steps:
            visit(step.step_id)

        return result

    @staticmethod
    def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
        """Extract a JSON array from text.

        Args:
            text: Raw text possibly containing a JSON array.

        Returns:
            Parsed list or None.
        """
        import re

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        patterns = [
            r"```json\s*\n(\[.*?\])\s*\n```",
            r"```\s*\n(\[.*?\])\s*\n```",
            r"(\[[\s\S]*\])",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from text.

        Args:
            text: Raw text possibly containing a JSON object.

        Returns:
            Parsed dict or None.
        """
        import re

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        patterns = [
            r"```json\s*\n(\{.*?\})\s*\n```",
            r"```\s*\n(\{.*?\})\s*\n```",
            r"(\{[\s\S]*\})",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
        return None
