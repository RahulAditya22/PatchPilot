"""Sandboxed test execution via subprocess with timeouts.

Runs tests in isolated subprocess with configurable timeout.
Supports pytest output parsing for structured results.

Model Attribution: Gemini (tool wiring)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

from patchpilot.config import get_config
from patchpilot.models import FailingTest, TestResult

logger = logging.getLogger(__name__)


class SandboxRunner:
    """Sandboxed test execution and command running.

    Uses subprocess with timeouts for isolated test execution.
    Parses pytest output into structured TestResult objects.
    """

    async def run_tests(
        self,
        repo_path: str,
        test_command: str | None = None,
        timeout: int | None = None,
    ) -> TestResult:
        """Run tests in the repo and return structured results.

        Args:
            repo_path: Path to the repository root.
            test_command: Override test command. Auto-detected if None.
            timeout: Timeout in seconds. Uses config default if None.

        Returns:
            Parsed TestResult with pass/fail counts and failure details.
        """
        if test_command is None:
            test_command = self._detect_test_command(repo_path)

        config = get_config()
        if timeout is None:
            timeout = config.SANDBOX_TIMEOUT

        logger.info("Running tests: %s (timeout=%ds)", test_command, timeout)
        start_time = time.monotonic()
        exit_code, stdout, stderr = await self.run_command(test_command, repo_path, timeout)
        duration = time.monotonic() - start_time

        if "pytest" in test_command:
            return self._parse_pytest_output(stdout, stderr, exit_code, duration)

        # Generic fallback for non-pytest test runners
        return TestResult(
            passed=exit_code == 0,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            num_passed=0,
            num_failed=0 if exit_code == 0 else 1,
            num_errors=0,
            failing_tests=[],
        )

    async def run_command(
        self,
        command: str,
        cwd: str,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """Run an arbitrary command in a subprocess.

        Args:
            command: Shell command to execute.
            cwd: Working directory for the command.
            timeout: Timeout in seconds. Uses config default if None.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        config = get_config()
        if timeout is None:
            timeout = config.SANDBOX_TIMEOUT

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            exit_code = process.returncode if process.returncode is not None else -1

            return exit_code, stdout, stderr

        except TimeoutError:
            logger.warning("Command timed out after %ds: %s", timeout, command)
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            return -1, "", f"Command timed out after {timeout} seconds."
        except Exception as exc:
            logger.error("Error running command: %s", exc)
            return -1, "", f"Error running command: {exc}"

    def _detect_test_command(self, repo_path: str) -> str:
        """Auto-detect the test command for a repository.

        Checks for common test framework config files.

        Args:
            repo_path: Path to the repository root.

        Returns:
            Detected test command string.
        """
        p = Path(repo_path)

        # Check for pytest indicators
        if (p / "pytest.ini").exists():
            return "python -m pytest -v"
        if (p / "conftest.py").exists():
            return "python -m pytest -v"

        # Check pyproject.toml for pytest config
        pyproject = p / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                if "pytest" in content or "tool.pytest" in content:
                    return "python -m pytest -v"
            except Exception:
                pass

        # Check setup.cfg
        if (p / "setup.cfg").exists():
            return "python -m pytest -v"

        # Check tox
        if (p / "tox.ini").exists():
            return "python -m pytest -v"

        # Default
        return "python -m pytest -v"

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration: float,
    ) -> TestResult:
        """Parse pytest output into a structured TestResult.

        Args:
            stdout: Standard output from pytest.
            stderr: Standard error from pytest.
            exit_code: Process exit code.
            duration: Wall-clock duration in seconds.

        Returns:
            Parsed TestResult with counts and failing test details.
        """
        num_passed = 0
        num_failed = 0
        num_errors = 0

        # Parse the summary line: "== 5 passed, 2 failed, 1 error in 1.23s =="
        combined = stdout + "\n" + stderr
        for line in combined.splitlines():
            if "==" in line and ("passed" in line or "failed" in line or "error" in line):
                matches = re.findall(r"(\d+)\s+(passed|failed|error)", line)
                for count_str, status in matches:
                    count = int(count_str)
                    if status == "passed":
                        num_passed = count
                    elif status == "failed":
                        num_failed = count
                    elif status == "error":
                        num_errors = count

        # Parse failing test names and tracebacks
        failing_tests = self._extract_failing_tests(combined)

        is_success = exit_code == 0 and num_failed == 0 and num_errors == 0

        return TestResult(
            passed=is_success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            num_passed=num_passed,
            num_failed=num_failed,
            num_errors=num_errors,
            failing_tests=failing_tests,
        )

    @staticmethod
    def _extract_failing_tests(output: str) -> list[FailingTest]:
        """Extract individual failing test details from pytest output.

        Args:
            output: Combined stdout+stderr from pytest.

        Returns:
            List of FailingTest objects with names and error messages.
        """
        failing_tests: list[FailingTest] = []

        # Match "FAILED tests/test_foo.py::test_bar - AssertionError: ..."
        failed_pattern = re.compile(r"FAILED\s+(\S+?)(?:\s+-\s+(.+))?$", re.MULTILINE)
        for match in failed_pattern.finditer(output):
            test_name = match.group(1)
            error_msg = match.group(2) or "Test failed"
            failing_tests.append(
                FailingTest(
                    test_name=test_name,
                    error_message=error_msg,
                    traceback_text="",
                )
            )

        # Also try to extract traceback sections
        # Pytest formats them as "_ test_name _" sections
        tb_pattern = re.compile(
            r"_{3,}\s+(\S+)\s+_{3,}\n(.*?)(?=_{3,}|={3,}|\Z)",
            re.DOTALL,
        )
        for match in tb_pattern.finditer(output):
            test_name = match.group(1)
            traceback_text = match.group(2).strip()

            # Find matching failing test and update traceback
            for ft in failing_tests:
                if test_name in ft.test_name:
                    # Create new FailingTest since it's frozen
                    idx = failing_tests.index(ft)
                    failing_tests[idx] = FailingTest(
                        test_name=ft.test_name,
                        error_message=ft.error_message,
                        traceback_text=traceback_text[:2000],
                    )
                    break

        return failing_tests
