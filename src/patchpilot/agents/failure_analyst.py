"""Failure Analyst Agent — diagnoses test/build failures and suggests fixes.

Parses test output, classifies the root cause, identifies the failing location,
and produces actionable fix suggestions for the Coding Agent.

Model Attribution: Claude Opus (complex diagnosis reasoning)
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from patchpilot.models import (
    CodeChange,
    Diagnosis,
    RootCause,
    TestResult,
)

if TYPE_CHECKING:
    from patchpilot.llm.client import LLMClient
    from patchpilot.observability.tracer import AgentTracer

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are an expert software debugging analyst. Your job is to analyze test/build
failures and produce precise diagnoses with actionable fix suggestions.

## Your Process:
1. Parse the error output carefully
2. Identify the root cause category
3. Locate the exact failing code
4. Suggest a concrete fix

## Root Cause Categories:
- SYNTAX_ERROR: Python syntax errors, indentation, missing colons, etc.
- LOGIC_ERROR: Code runs but produces wrong results
- IMPORT_ERROR: Missing imports, circular imports, wrong module paths
- TEST_ASSERTION: Test assertions fail (expected vs actual mismatch)
- RUNTIME_CRASH: Unhandled exceptions during execution
- TIMEOUT: Test/process exceeded time limit
- UNKNOWN: Cannot determine root cause

## Output Format:
Respond with a JSON object:
```json
{
    "root_cause": "SYNTAX_ERROR|LOGIC_ERROR|IMPORT_ERROR|TEST_ASSERTION|RUNTIME_CRASH|TIMEOUT|UNKNOWN",
    "explanation": "Detailed explanation of what went wrong",
    "confidence": 0.95,
    "suggested_fixes": [
        {
            "file_path": "path/to/file.py",
            "new_content": "complete corrected file content",
            "change_description": "what the fix does"
        }
    ]
}
```

Be precise. Reference exact line numbers and error messages. Your fixes must be concrete
and complete — no placeholders or partial solutions."""


class FailureAnalyst:
    """Agent that diagnoses test/build failures and suggests fixes.

    Combines heuristic pre-analysis (regex pattern matching on common errors)
    with LLM-powered deep analysis for complex failures.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tracer: AgentTracer | None = None,
    ) -> None:
        """Initialize the failure analyst.

        Args:
            llm_client: LLM client for analysis.
            tracer: Optional tracer for observability.
        """
        self._llm = llm_client
        self._tracer = tracer

    async def analyze(
        self,
        test_result: TestResult,
        changes: list[CodeChange],
        file_contents: dict[str, str] | None = None,
    ) -> Diagnosis:
        """Analyze a test failure and produce a diagnosis.

        Args:
            test_result: The failed test result.
            changes: Code changes that were applied before the test run.
            file_contents: Optional map of file paths to current contents.

        Returns:
            Diagnosis with root cause, explanation, and suggested fixes.
        """
        step_id = None
        if self._tracer:
            step_id = self._tracer.start_step(
                "failure_analysis",
                input_summary=f"Analyzing {test_result.num_failed} failures",
            )

        try:
            # Step 1: Heuristic pre-analysis
            heuristic = self._heuristic_analysis(test_result)
            logger.info(
                "Heuristic pre-analysis: %s (confidence: %.2f)",
                heuristic.root_cause.value,
                heuristic.confidence,
            )

            # Step 2: If heuristic is highly confident and simple, skip LLM
            if heuristic.confidence >= 0.95 and heuristic.root_cause in (
                RootCause.SYNTAX_ERROR,
                RootCause.IMPORT_ERROR,
            ):
                logger.info("High-confidence heuristic diagnosis, skipping LLM analysis")
                return heuristic

            # Step 3: LLM-powered deep analysis
            diagnosis = await self._llm_analysis(test_result, changes, file_contents, heuristic)
            return diagnosis

        finally:
            if self._tracer and step_id:
                self._tracer.end_step(
                    step_id,
                    output_summary=(
                        f"Diagnosis: {diagnosis.root_cause.value}"
                        if "diagnosis" in dir()
                        else "Analysis incomplete"
                    ),
                    model_used=self._llm.model,
                )

    def _heuristic_analysis(self, test_result: TestResult) -> Diagnosis:
        """Perform fast heuristic analysis using regex patterns.

        Args:
            test_result: The failed test result.

        Returns:
            A preliminary Diagnosis based on pattern matching.
        """
        combined_output = f"{test_result.stdout}\n{test_result.stderr}"

        # Syntax error patterns
        syntax_patterns = [
            r"SyntaxError: (.+)",
            r"IndentationError: (.+)",
            r"TabError: (.+)",
        ]
        for pattern in syntax_patterns:
            match = re.search(pattern, combined_output)
            if match:
                return Diagnosis(
                    root_cause=RootCause.SYNTAX_ERROR,
                    explanation=f"Python syntax error: {match.group(1)}",
                    suggested_fixes=[],
                    confidence=0.95,
                )

        # Import error patterns
        import_patterns = [
            r"ImportError: (.+)",
            r"ModuleNotFoundError: (.+)",
            r"cannot import name '(\w+)'",
        ]
        for pattern in import_patterns:
            match = re.search(pattern, combined_output)
            if match:
                return Diagnosis(
                    root_cause=RootCause.IMPORT_ERROR,
                    explanation=f"Import error: {match.group(1)}",
                    suggested_fixes=[],
                    confidence=0.90,
                )

        # Assertion error patterns
        assertion_patterns = [
            r"AssertionError: (.+)",
            r"assert .+ == .+",
            r"FAILED .+ - assert",
            r"E\s+assert\s+",
        ]
        for pattern in assertion_patterns:
            if re.search(pattern, combined_output):
                return Diagnosis(
                    root_cause=RootCause.TEST_ASSERTION,
                    explanation="Test assertion failure detected",
                    suggested_fixes=[],
                    confidence=0.80,
                )

        # Timeout patterns
        timeout_patterns = [
            r"TimeoutError",
            r"timed out",
            r"TIMEOUT",
        ]
        for pattern in timeout_patterns:
            if re.search(pattern, combined_output, re.IGNORECASE):
                return Diagnosis(
                    root_cause=RootCause.TIMEOUT,
                    explanation="Test execution timed out",
                    suggested_fixes=[],
                    confidence=0.90,
                )

        # Runtime crash patterns
        crash_patterns = [
            r"TypeError: (.+)",
            r"ValueError: (.+)",
            r"AttributeError: (.+)",
            r"KeyError: (.+)",
            r"RuntimeError: (.+)",
            r"NameError: (.+)",
        ]
        for pattern in crash_patterns:
            match = re.search(pattern, combined_output)
            if match:
                return Diagnosis(
                    root_cause=RootCause.RUNTIME_CRASH,
                    explanation=f"Runtime error: {match.group(0)}",
                    suggested_fixes=[],
                    confidence=0.75,
                )

        return Diagnosis(
            root_cause=RootCause.UNKNOWN,
            explanation="Could not determine root cause from heuristics",
            suggested_fixes=[],
            confidence=0.1,
        )

    async def _llm_analysis(
        self,
        test_result: TestResult,
        changes: list[CodeChange],
        file_contents: dict[str, str] | None,
        heuristic: Diagnosis,
    ) -> Diagnosis:
        """Perform deep LLM-powered analysis.

        Args:
            test_result: Failed test result.
            changes: Code changes that were applied.
            file_contents: Current file contents.
            heuristic: Preliminary heuristic diagnosis.

        Returns:
            Full Diagnosis with suggested fixes.
        """
        # Build context about what changed
        changes_summary = "\n".join(
            f"### {c.file_path}\n**Change:** {c.change_description}\n"
            f"```diff\n{self._simple_diff(c.original_content, c.modified_content)}\n```"
            for c in changes[:5]  # Cap at 5 most recent changes
        )

        # Build failing test details
        failing_details = ""
        for ft in test_result.failing_tests[:10]:  # Cap at 10 failures
            failing_details += f"\n**Test:** {ft.test_name}\n"
            failing_details += f"**Error:** {ft.error_message}\n"
            if ft.traceback_text:
                failing_details += f"```\n{ft.traceback_text[:1000]}\n```\n"

        # Include relevant file contents
        files_section = ""
        if file_contents:
            for path, content in list(file_contents.items())[:3]:
                files_section += f"\n### {path}\n```python\n{content[:3000]}\n```\n"

        messages = [
            {
                "role": "user",
                "content": f"""## Test Failure Analysis

### Test Output
**Exit Code:** {test_result.exit_code}
**Passed:** {test_result.num_passed} | **Failed:** {test_result.num_failed} | **Errors:** {test_result.num_errors}

### Stderr
```
{test_result.stderr[:3000]}
```

### Stdout
```
{test_result.stdout[:3000]}
```

### Failing Tests
{failing_details}

### Recent Code Changes
{changes_summary}

### Relevant File Contents
{files_section}

### Heuristic Pre-Analysis
**Root Cause Guess:** {heuristic.root_cause.value}
**Explanation:** {heuristic.explanation}
**Confidence:** {heuristic.confidence}

## Instructions
Analyze these failures deeply. Identify the root cause and provide concrete fixes.
For each fix, provide the COMPLETE corrected file content (not just the changed lines).
Return your analysis as JSON.""",
            }
        ]

        response = await self._llm.generate(
            messages=messages,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=8192,
        )

        return self._parse_diagnosis(response.content)

    def _parse_diagnosis(self, content: str) -> Diagnosis:
        """Parse LLM response into a Diagnosis.

        Args:
            content: Raw LLM response.

        Returns:
            Parsed Diagnosis.
        """
        json_data = self._extract_json(content)

        if json_data is None:
            logger.warning("Could not parse diagnosis JSON, returning heuristic-level diagnosis")
            return Diagnosis(
                root_cause=RootCause.UNKNOWN,
                explanation=content[:500],
                suggested_fixes=[],
                confidence=0.3,
            )

        try:
            root_cause_str = json_data.get("root_cause", "UNKNOWN").upper()
            try:
                root_cause = RootCause[root_cause_str]
            except KeyError:
                root_cause = RootCause.UNKNOWN

            fixes: list[CodeChange] = []
            for fix_data in json_data.get("suggested_fixes", []):
                fixes.append(
                    CodeChange(
                        file_path=fix_data.get("file_path", ""),
                        original_content=fix_data.get("original_content", ""),
                        modified_content=fix_data.get("new_content", ""),
                        change_description=fix_data.get("change_description", ""),
                    )
                )

            return Diagnosis(
                root_cause=root_cause,
                explanation=json_data.get("explanation", ""),
                suggested_fixes=fixes,
                confidence=float(json_data.get("confidence", 0.5)),
            )
        except Exception as exc:
            logger.error("Failed to parse diagnosis: %s", exc)
            return Diagnosis(
                root_cause=RootCause.UNKNOWN,
                explanation=f"Parse error: {exc}\nRaw: {content[:300]}",
                suggested_fixes=[],
                confidence=0.1,
            )

    @staticmethod
    def _simple_diff(original: str, modified: str) -> str:
        """Create a simple unified diff between two strings.

        Args:
            original: Original content.
            modified: Modified content.

        Returns:
            Unified diff string.
        """
        import difflib

        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile="original",
            tofile="modified",
            lineterm="",
        )
        result = "\n".join(diff)
        # Truncate very long diffs
        if len(result) > 2000:
            result = result[:2000] + "\n... (truncated)"
        return result

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from text.

        Args:
            text: Raw text possibly containing JSON.

        Returns:
            Parsed dict or None.
        """
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        patterns = [
            r"```json\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
            r"\{[\s\S]*\}",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    continue
        return None
