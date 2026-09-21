from __future__ import annotations

import pytest
from pydantic import ValidationError

from patchpilot.models import (
    ActionType,
    AgentResult,
    CodeChange,
    Complexity,
    IssueRequirement,
    PlanStep,
    RootCause,
    Severity,
)


def test_issue_requirement_creation() -> None:
    """Create IssueRequirement, verify fields."""
    req = IssueRequirement(
        title="Fix issue",
        description="Detailed description",
        requirements=["req1"],
        labels=["bug"],
        priority="high",
    )
    assert req.title == "Fix issue"
    assert req.description == "Detailed description"
    assert req.requirements == ["req1"]


def test_plan_step_creation() -> None:
    """Create PlanStep with all fields."""
    step = PlanStep(
        step_id="step-1",
        description="Change a file",
        target_files=["test.py"],
        action_type=ActionType.MODIFY,
        rationale="needed",
        dependencies=[],
        estimated_complexity=Complexity.LOW,
    )
    assert step.step_id == "step-1"
    assert step.action_type == ActionType.MODIFY
    assert step.target_files == ["test.py"]


def test_code_change_frozen() -> None:
    """Verify frozen model raises on mutation."""
    change = CodeChange(
        file_path="test.py",
        original_content="old",
        modified_content="new",
        change_description="update",
    )
    with pytest.raises(ValidationError):
        change.file_path = "other.py"  # type: ignore[misc]


def test_action_type_enum_values() -> None:
    """Verify MODIFY, CREATE, DELETE exist."""
    assert ActionType.MODIFY.value == "MODIFY"
    assert ActionType.CREATE.value == "CREATE"
    assert ActionType.DELETE.value == "DELETE"


def test_root_cause_enum_values() -> None:
    """Verify RootCause values."""
    assert RootCause.SYNTAX_ERROR.value == "SYNTAX_ERROR"
    assert RootCause.LOGIC_ERROR.value == "LOGIC_ERROR"
    assert RootCause.IMPORT_ERROR.value == "IMPORT_ERROR"
    assert RootCause.TEST_ASSERTION.value == "TEST_ASSERTION"


def test_severity_enum_values() -> None:
    """Verify LOW, MEDIUM, HIGH, CRITICAL."""
    assert Severity.LOW.value == "LOW"
    assert Severity.MEDIUM.value == "MEDIUM"
    assert Severity.HIGH.value == "HIGH"
    assert Severity.CRITICAL.value == "CRITICAL"


def test_agent_result_with_defaults() -> None:
    """Create with minimal fields."""
    result = AgentResult(success=True)
    assert result.success is True
    assert result.error_message is None
