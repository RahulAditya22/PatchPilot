"""Orchestrator Agent — main control loop for the PatchPilot pipeline.

Coordinates the full flow: repo ingestion → indexing → planning → coding →
testing → failure analysis → retry → security checks → report generation.

Model Attribution: Claude Opus (critical control flow)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from patchpilot.config import get_config
from patchpilot.models import (
    AgentResult,
    CodeChange,
    IssueRequirement,
    Plan,
    SecurityReport,
    TestResult,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestration engine for the PatchPilot pipeline.

    Manages the full lifecycle of processing a GitHub issue:
    1. Ingest the repository
    2. Index the codebase for search
    3. Parse the issue into requirements
    4. Generate an implementation plan
    5. Apply code changes
    6. Run tests in sandbox
    7. Analyze failures and retry (bounded loop)
    8. Run security checks
    9. Generate diff and report

    Each step is traced for observability.
    """

    def __init__(self) -> None:
        """Initialize the orchestrator with all required components."""
        self._config = get_config()
        self._components_initialized = False

        # Lazy-initialized components (set up in _init_components)
        self._planner: Any = None
        self._coder: Any = None
        self._analyst: Any = None
        self._repo_ingester: Any = None
        self._indexer: Any = None
        self._code_search: Any = None
        self._issue_parser: Any = None
        self._sandbox: Any = None
        self._diff_generator: Any = None
        self._security_checker: Any = None
        self._dep_analyzer: Any = None
        self._tracer: Any = None
        self._token_tracker: Any = None

    async def _init_components(self) -> None:
        """Lazily initialize all pipeline components.

        This is separate from __init__ to allow async setup and
        to defer heavy imports until actually needed.
        """
        if self._components_initialized:
            return

        from patchpilot.agents.coder import CodingAgent
        from patchpilot.agents.failure_analyst import FailureAnalyst
        from patchpilot.agents.planner import PlanningAgent
        from patchpilot.llm.client import LLMClient, LLMProvider
        from patchpilot.llm.token_tracker import get_tracker
        from patchpilot.observability.tracer import get_tracer
        from patchpilot.tools.code_search import CodeSearchEngine
        from patchpilot.tools.dependency_analyzer import DependencyAnalyzer
        from patchpilot.tools.diff_generator import DiffGenerator
        from patchpilot.tools.indexer import CodebaseIndexer
        from patchpilot.tools.issue_parser import IssueParser
        from patchpilot.tools.repo_ingester import RepoIngester
        from patchpilot.tools.sandbox import SandboxRunner
        from patchpilot.tools.security import SecurityChecker

        self._tracer = get_tracer()
        self._token_tracker = get_tracker()

        # Create LLM clients
        opus_client = LLMClient(
            provider=LLMProvider.ANTHROPIC,
            model=self._config.DEFAULT_OPUS_MODEL,
            api_key=self._config.ANTHROPIC_API_KEY,
        )

        # Initialize agents with Opus client (core reasoning)
        self._planner = PlanningAgent(opus_client, tracer=self._tracer)
        self._coder = CodingAgent(opus_client, tracer=self._tracer)
        self._analyst = FailureAnalyst(opus_client, tracer=self._tracer)

        # Initialize tools
        self._repo_ingester = RepoIngester()
        self._indexer = CodebaseIndexer()
        self._code_search = CodeSearchEngine()
        self._issue_parser = IssueParser()
        self._sandbox = SandboxRunner()
        self._diff_generator = DiffGenerator()
        self._security_checker = SecurityChecker()
        self._dep_analyzer = DependencyAnalyzer()

        self._components_initialized = True
        logger.info("All orchestrator components initialized")

    async def run(
        self,
        repo_url: str,
        issue_text: str,
        issue_title: str = "",
        max_retries: int | None = None,
    ) -> AgentResult:
        """Run the full PatchPilot pipeline.

        Args:
            repo_url: URL of the GitHub repository to work on.
            issue_text: Text of the GitHub issue or feature request.
            issue_title: Optional title of the issue.
            max_retries: Max retry attempts for test failures. Defaults to config.

        Returns:
            AgentResult with all changes, test results, diff, and report.
        """
        await self._init_components()

        if max_retries is None:
            max_retries = self._config.MAX_RETRIES

        run_step_id = self._tracer.start_step(
            "orchestrator_run",
            input_summary=f"Repo: {repo_url}, Issue: {issue_title or issue_text[:80]}",
        )

        try:
            result = await self._execute_pipeline(repo_url, issue_text, issue_title, max_retries)
            return result
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return AgentResult(
                success=False,
                plan=Plan(issue_summary="", steps=[], reasoning=""),
                changes=[],
                test_results=[],
                security_report=SecurityReport(passed=False, findings=[]),
                diff_text="",
                report_markdown=f"# Pipeline Failed\n\nError: {exc}",
                total_token_usage=self._token_tracker.get_all_usage(),
                traces=self._tracer.get_traces(),
                error_message=str(exc),
            )
        finally:
            self._tracer.end_step(
                run_step_id,
                output_summary="Pipeline completed",
            )

    async def _execute_pipeline(
        self,
        repo_url: str,
        issue_text: str,
        issue_title: str,
        max_retries: int,
    ) -> AgentResult:
        """Execute the full pipeline with error handling at each stage.

        Args:
            repo_url: Repository URL.
            issue_text: Issue body text.
            issue_title: Issue title.
            max_retries: Max retry count.

        Returns:
            Complete AgentResult.
        """
        # === Stage 1: Ingest Repository ===
        logger.info("Stage 1: Ingesting repository %s", repo_url)
        step_id = self._tracer.start_step("repo_ingestion", input_summary=repo_url)
        repo_info = await self._repo_ingester.ingest(repo_url)
        self._tracer.end_step(step_id, output_summary=f"Cloned to {repo_info.clone_path}")

        # === Stage 2: Index Codebase ===
        logger.info("Stage 2: Indexing codebase")
        step_id = self._tracer.start_step("codebase_indexing")
        await self._indexer.index_repository(repo_info.clone_path)
        self._tracer.end_step(step_id, output_summary="Indexing complete")

        # === Stage 3: Parse Issue ===
        logger.info("Stage 3: Parsing issue")
        step_id = self._tracer.start_step("issue_parsing", input_summary=issue_text[:200])
        issue = await self._issue_parser.parse(
            title=issue_title or "Issue",
            body=issue_text,
        )
        self._tracer.end_step(
            step_id,
            output_summary=f"Requirements: {len(issue.requirements)}",
        )

        # === Stage 4: Search for Context ===
        logger.info("Stage 4: Searching codebase for relevant context")
        step_id = self._tracer.start_step("context_search")
        context = await self._code_search.search(
            query=f"{issue.title} {issue.description}",
            search_type="hybrid",
            max_results=20,
        )
        context_text = "\n\n".join(
            f"### {r['file_path']} (score: {r['score']:.2f})\n```\n{r['content'][:500]}\n```"
            for r in context
        )
        self._tracer.end_step(
            step_id,
            output_summary=f"Found {len(context)} relevant code snippets",
        )

        # Register tool handlers for the planner
        self._register_planner_tools(repo_info.clone_path)

        # === Stage 5: Create Plan ===
        logger.info("Stage 5: Creating implementation plan")
        plan = await self._planner.create_plan(
            issue=issue,
            codebase_context=context_text,
            file_tree=repo_info.file_tree,
        )
        logger.info("Plan created with %d steps", len(plan.steps))

        # Set up file reader for coder
        self._coder.set_file_reader(self._make_file_reader(repo_info.clone_path))

        # === Stage 6: Apply Changes ===
        logger.info("Stage 6: Applying code changes")
        changes = await self._coder.apply_plan(plan, repo_info.clone_path)
        logger.info("Applied %d code changes", len(changes))

        # Write changes to disk
        self._write_changes(changes, repo_info.clone_path)

        # === Stage 7: Test + Retry Loop ===
        logger.info("Stage 7: Running tests (max %d retries)", max_retries)
        test_results: list[TestResult] = []
        final_test_result: TestResult | None = None

        for attempt in range(max_retries + 1):
            step_id = self._tracer.start_step(
                f"test_attempt_{attempt}",
                input_summary=f"Attempt {attempt + 1}/{max_retries + 1}",
            )

            test_result = await self._sandbox.run_tests(repo_info.clone_path)
            test_results.append(test_result)
            final_test_result = test_result

            self._tracer.end_step(
                step_id,
                output_summary=(
                    f"Passed: {test_result.num_passed}, Failed: {test_result.num_failed}"
                ),
            )

            if test_result.passed:
                logger.info("Tests passed on attempt %d", attempt + 1)
                break

            if attempt < max_retries:
                logger.info(
                    "Tests failed (attempt %d/%d), analyzing and retrying",
                    attempt + 1,
                    max_retries + 1,
                )

                # Get current file contents for analysis
                file_contents = self._read_changed_files(changes, repo_info.clone_path)

                # Analyze failure
                diagnosis = await self._analyst.analyze(test_result, changes, file_contents)
                logger.info(
                    "Diagnosis: %s (confidence: %.2f)",
                    diagnosis.root_cause.value,
                    diagnosis.confidence,
                )

                # Apply fixes
                if diagnosis.suggested_fixes:
                    for fix in diagnosis.suggested_fixes:
                        changes.append(fix)
                    self._write_changes(diagnosis.suggested_fixes, repo_info.clone_path)
                else:
                    # Ask coder to fix based on diagnosis
                    for change in changes:
                        if any(
                            change.file_path in ft.test_name for ft in test_result.failing_tests
                        ):
                            fix = await self._coder.apply_fix(
                                file_path=change.file_path,
                                current_content=change.modified_content,
                                fix_description=diagnosis.explanation,
                                error_context=test_result.stderr[:2000],
                            )
                            changes.append(fix)
                            self._write_changes([fix], repo_info.clone_path)
                            break
            else:
                logger.warning("Max retries exhausted, proceeding with failures")

        # === Stage 8: Security Checks ===
        logger.info("Stage 8: Running security checks")
        step_id = self._tracer.start_step("security_checks")
        security_report = await self._security_checker.check(changes)
        self._tracer.end_step(
            step_id,
            output_summary=f"Passed: {security_report.passed}, Findings: {len(security_report.findings)}",
        )

        # === Stage 9: Generate Diff and Report ===
        logger.info("Stage 9: Generating diff and report")
        diff_text = self._diff_generator.generate_unified_diff(changes)
        report = self._generate_report(
            plan=plan,
            changes=changes,
            test_results=test_results,
            security_report=security_report,
            issue=issue,
        )

        success = (
            final_test_result is not None and final_test_result.passed
        ) and security_report.passed

        return AgentResult(
            success=success,
            plan=plan,
            changes=changes,
            test_results=test_results,
            security_report=security_report,
            diff_text=diff_text,
            report_markdown=report,
            total_token_usage=self._token_tracker.get_all_usage(),
            traces=self._tracer.get_traces(),
            error_message=None if success else "Tests or security checks did not pass",
        )

    def _register_planner_tools(self, repo_root: str) -> None:
        """Register tool handlers for the planning agent.

        Args:
            repo_root: Root directory of the cloned repository.
        """

        async def search_code(
            query: str,
            search_type: str = "hybrid",
            max_results: int = 10,
        ) -> str:
            results = await self._code_search.search(
                query=query,
                search_type=search_type,
                max_results=max_results,
            )
            return "\n\n".join(
                f"**{r['file_path']}** (score: {r['score']:.2f}):\n{r['content'][:300]}"
                for r in results
            )

        async def read_file(file_path: str) -> str:
            full_path = os.path.join(repo_root, file_path)
            try:
                with open(full_path, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except FileNotFoundError:
                return f"File not found: {file_path}"

        async def list_files(directory: str = ".", pattern: str = "*") -> str:

            dir_path = Path(repo_root) / directory
            if not dir_path.exists():
                return f"Directory not found: {directory}"
            files = []
            for p in dir_path.rglob(pattern):
                rel = p.relative_to(repo_root)
                files.append(str(rel))
            return "\n".join(files[:200])

        async def analyze_dependencies(file_path: str) -> str:
            deps = await self._dep_analyzer.analyze(os.path.join(repo_root, file_path))
            return "\n".join(f"- {d}" for d in deps)

        self._planner.register_tool_handler("search_code", search_code)
        self._planner.register_tool_handler("read_file", read_file)
        self._planner.register_tool_handler("list_files", list_files)
        self._planner.register_tool_handler("analyze_dependencies", analyze_dependencies)

    def _make_file_reader(self, repo_root: str) -> Any:
        """Create an async file reader closure for the coding agent.

        Args:
            repo_root: Root of the repository.

        Returns:
            Async callable that reads file contents.
        """

        async def read_file(file_path: str) -> str:
            full_path = os.path.join(repo_root, file_path)
            with open(full_path, encoding="utf-8", errors="replace") as f:
                return f.read()

        return read_file

    @staticmethod
    def _write_changes(changes: list[CodeChange], repo_root: str) -> None:
        """Write code changes to disk.

        Args:
            changes: List of changes to write.
            repo_root: Root directory of the repository.
        """
        for change in changes:
            file_path = os.path.join(repo_root, change.file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(change.modified_content)
            logger.debug("Wrote change to %s", change.file_path)

    @staticmethod
    def _read_changed_files(changes: list[CodeChange], repo_root: str) -> dict[str, str]:
        """Read current contents of changed files from disk.

        Args:
            changes: List of changes (for file paths).
            repo_root: Repository root.

        Returns:
            Map of file paths to current contents.
        """
        contents: dict[str, str] = {}
        seen_paths: set[str] = set()
        for change in changes:
            if change.file_path in seen_paths:
                continue
            seen_paths.add(change.file_path)
            full_path = os.path.join(repo_root, change.file_path)
            try:
                with open(full_path, encoding="utf-8", errors="replace") as f:
                    contents[change.file_path] = f.read()
            except FileNotFoundError:
                contents[change.file_path] = ""
        return contents

    @staticmethod
    def _generate_report(
        plan: Plan,
        changes: list[CodeChange],
        test_results: list[TestResult],
        security_report: SecurityReport,
        issue: IssueRequirement,
    ) -> str:
        """Generate a PR-style markdown report.

        Args:
            plan: The implementation plan.
            changes: All code changes.
            test_results: Test run results.
            security_report: Security check results.
            issue: Original issue.

        Returns:
            Formatted markdown report.
        """
        # Determine overall status
        final_test = test_results[-1] if test_results else None
        tests_passed = final_test.passed if final_test else False
        overall_status = "✅ Success" if (tests_passed and security_report.passed) else "❌ Failed"

        report_parts = [
            f"# PatchPilot Report: {issue.title}",
            f"\n**Status:** {overall_status}\n",
            "## Issue Summary",
            plan.issue_summary,
            "\n## Implementation Plan",
            f"**Steps:** {len(plan.steps)}\n",
        ]

        for step in plan.steps:
            report_parts.append(
                f"- **{step.step_id}** [{step.action_type.value}] "
                f"{step.description} → `{', '.join(step.target_files)}`"
            )

        report_parts.append("\n## Files Changed")
        seen_files: set[str] = set()
        for change in changes:
            if change.file_path not in seen_files:
                seen_files.add(change.file_path)
                report_parts.append(f"- `{change.file_path}`: {change.change_description}")

        report_parts.append("\n## Test Results")
        for i, tr in enumerate(test_results):
            status_emoji = "✅" if tr.passed else "❌"
            report_parts.append(
                f"- Attempt {i + 1}: {status_emoji} "
                f"Passed={tr.num_passed} Failed={tr.num_failed} "
                f"Errors={tr.num_errors} ({tr.duration_seconds:.1f}s)"
            )

        report_parts.append("\n## Security Report")
        sec_status = "✅ Passed" if security_report.passed else "❌ Issues Found"
        report_parts.append(f"**Status:** {sec_status}")
        for finding in security_report.findings:
            report_parts.append(
                f"- [{finding.severity.value}] {finding.finding_type}: "
                f"{finding.description} ({finding.file_path})"
            )

        return "\n".join(report_parts)
