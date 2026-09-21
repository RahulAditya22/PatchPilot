"""Shared test fixtures for PatchPilot test suite."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from patchpilot.config import Config
from patchpilot.models import (
    ActionType,
    CodeChange,
    Complexity,
    FailingTest,
    IssueRequirement,
    Plan,
    PlanStep,
    TestResult,
    TokenUsage,
)


@pytest.fixture
def sample_issue() -> IssueRequirement:
    """Create a sample issue requirement for testing."""
    return IssueRequirement(
        title="Add input validation to user registration",
        description="The user registration endpoint accepts any input without validation. "
        "We need to add email format validation and password strength checking.",
        requirements=[
            "Validate email format using regex",
            "Require passwords to be at least 8 characters",
            "Return descriptive error messages for invalid input",
        ],
        labels=["bug", "security"],
        priority="high",
    )


@pytest.fixture
def sample_plan() -> Plan:
    """Create a sample implementation plan for testing."""
    return Plan(
        issue_summary="Add input validation to user registration endpoint",
        steps=[
            PlanStep(
                step_id="step_1",
                description="Add email validation function",
                target_files=["src/validators.py"],
                action_type=ActionType.MODIFY,
                rationale="Need a reusable email validation function",
                dependencies=[],
                estimated_complexity=Complexity.LOW,
            ),
            PlanStep(
                step_id="step_2",
                description="Add password strength validator",
                target_files=["src/validators.py"],
                action_type=ActionType.MODIFY,
                rationale="Password validation per requirements",
                dependencies=["step_1"],
                estimated_complexity=Complexity.LOW,
            ),
            PlanStep(
                step_id="step_3",
                description="Update registration endpoint to use validators",
                target_files=["src/api/routes.py"],
                action_type=ActionType.MODIFY,
                rationale="Integrate validators into the endpoint",
                dependencies=["step_1", "step_2"],
                estimated_complexity=Complexity.MEDIUM,
            ),
            PlanStep(
                step_id="step_4",
                description="Add tests for validators",
                target_files=["tests/test_validators.py"],
                action_type=ActionType.CREATE,
                rationale="Ensure validation logic works correctly",
                dependencies=["step_1", "step_2"],
                estimated_complexity=Complexity.LOW,
            ),
        ],
        reasoning="The issue requires adding validation. Best approach is to create "
        "a validators module, then integrate into the existing endpoint.",
    )


@pytest.fixture
def sample_code_change() -> CodeChange:
    """Create a sample code change for testing."""
    return CodeChange(
        file_path="src/validators.py",
        original_content="def validate_email(email):\n    return True\n",
        modified_content="import re\n\ndef validate_email(email: str) -> bool:\n"
        '    """Validate email format."""\n'
        '    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"\n'
        "    return bool(re.match(pattern, email))\n",
        change_description="Added proper email validation with regex",
    )


@pytest.fixture
def sample_test_result_pass() -> TestResult:
    """Create a passing test result."""
    return TestResult(
        passed=True,
        exit_code=0,
        stdout="===== 5 passed in 0.3s =====",
        stderr="",
        duration_seconds=0.3,
        num_passed=5,
        num_failed=0,
        num_errors=0,
        failing_tests=[],
    )


@pytest.fixture
def sample_test_result_fail() -> TestResult:
    """Create a failing test result."""
    return TestResult(
        passed=False,
        exit_code=1,
        stdout="FAILED tests/test_validators.py::test_email_validation\n"
        "===== 1 failed, 4 passed in 0.4s =====",
        stderr="E       AssertionError: assert False == True\n"
        'E       where False = validate_email("invalid")',
        duration_seconds=0.4,
        num_passed=4,
        num_failed=1,
        num_errors=0,
        failing_tests=[
            FailingTest(
                test_name="tests/test_validators.py::test_email_validation",
                error_message="AssertionError: assert False == True",
                traceback_text='File "tests/test_validators.py", line 10\n'
                "    assert validate_email('invalid') == True",
            )
        ],
    )


@pytest.fixture
def sample_token_usage() -> TokenUsage:
    """Create a sample token usage record."""
    from datetime import datetime

    return TokenUsage(
        model="claude-sonnet-4-20250514",
        input_tokens=1500,
        output_tokens=500,
        estimated_cost_usd=0.012,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for testing agents."""
    from patchpilot.llm.client import LLMResponse

    mock = MagicMock()
    mock.model = "claude-sonnet-4-20250514"
    mock.generate = AsyncMock(
        return_value=LLMResponse(
            content='{"test": "response"}',
            tool_calls=None,
            token_usage=TokenUsage(
                model="claude-sonnet-4-20250514",
                input_tokens=100,
                output_tokens=50,
                estimated_cost_usd=0.001,
            ),
            raw_response=None,
        )
    )
    return mock


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository structure for testing."""
    # Create a basic Python project structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "main.py").write_text(
        'def hello():\n    return "Hello, World!"\n',
        encoding="utf-8",
    )
    (src_dir / "validators.py").write_text(
        "def validate_email(email):\n    return True\n",
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_main.py").write_text(
        'from src.main import hello\n\ndef test_hello():\n    assert hello() == "Hello, World!"\n',
        encoding="utf-8",
    )

    # Create pyproject.toml
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test-project"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def config() -> Config:
    """Create a test config with no real API keys."""
    return Config(
        ANTHROPIC_API_KEY="test-key-not-real",
        GOOGLE_API_KEY="test-key-not-real",
        GITHUB_TOKEN="test-token-not-real",
        MAX_RETRIES=2,
        SANDBOX_TIMEOUT=30,
    )
