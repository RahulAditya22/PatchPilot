from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IssueRequirement(BaseModel):
    """Represents a parsed issue requirement."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Title of the issue")
    description: str = Field(description="Detailed description of the issue")
    requirements: list[str] = Field(
        default_factory=list, description="List of specific requirements extracted from the issue"
    )
    labels: list[str] = Field(default_factory=list, description="Labels associated with the issue")
    priority: str = Field(default="medium", description="Priority of the issue")


class ActionType(str, Enum):
    """Types of actions that can be performed in a plan step."""

    MODIFY = "MODIFY"
    CREATE = "CREATE"
    DELETE = "DELETE"


class Complexity(str, Enum):
    """Estimated complexity of a plan step."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PlanStep(BaseModel):
    """A single step in a plan."""

    model_config = ConfigDict(frozen=True)

    step_id: str = Field(description="Unique identifier for the step")
    description: str = Field(description="Description of the work to be done")
    target_files: list[str] = Field(description="List of files to be modified, created, or deleted")
    action_type: ActionType = Field(description="The type of action to perform")
    rationale: str = Field(description="Reasoning behind this step")
    dependencies: list[str] = Field(
        default_factory=list, description="List of step_ids this step depends on"
    )
    estimated_complexity: Complexity = Field(description="Estimated complexity of this step")


class Plan(BaseModel):
    """A full plan to address an issue."""

    model_config = ConfigDict(frozen=True)

    issue_summary: str = Field(description="Summary of the issue being addressed")
    steps: list[PlanStep] = Field(description="Ordered list of steps to complete the plan")
    reasoning: str = Field(description="Overall reasoning for the plan")


class CodeChange(BaseModel):
    """Represents a modification to a file."""

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(description="Path to the file being changed")
    original_content: str = Field(description="Original content of the file or section")
    modified_content: str = Field(description="New content to replace the original")
    change_description: str = Field(description="Description of the change")


class FailingTest(BaseModel):
    """Represents a single failing test."""

    model_config = ConfigDict(frozen=True)

    test_name: str = Field(description="Name of the test that failed")
    error_message: str = Field(description="The error message produced by the test")
    traceback_text: str = Field(description="The full traceback of the failure")


class TestResult(BaseModel):
    """Result of test execution."""

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(description="Whether all tests passed")
    exit_code: int = Field(description="Exit code of the test runner")
    stdout: str = Field(description="Standard output from the test runner")
    stderr: str = Field(description="Standard error from the test runner")
    duration_seconds: float = Field(description="Duration of the test run in seconds")
    num_passed: int = Field(description="Number of passing tests")
    num_failed: int = Field(description="Number of failing tests")
    num_errors: int = Field(description="Number of tests with errors")
    failing_tests: list[FailingTest] = Field(
        default_factory=list, description="List of tests that failed or errored"
    )


class RootCause(str, Enum):
    """Types of root causes for test failures or bugs."""

    SYNTAX_ERROR = "SYNTAX_ERROR"
    LOGIC_ERROR = "LOGIC_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    TEST_ASSERTION = "TEST_ASSERTION"
    RUNTIME_CRASH = "RUNTIME_CRASH"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class Diagnosis(BaseModel):
    """Analysis of a failure."""

    model_config = ConfigDict(frozen=True)

    root_cause: RootCause = Field(description="Identified root cause")
    explanation: str = Field(description="Detailed explanation of the issue")
    suggested_fixes: list[CodeChange] = Field(
        default_factory=list, description="Suggested code changes to fix the issue"
    )
    confidence: float = Field(
        description="Confidence score in the diagnosis (0.0 to 1.0)", ge=0.0, le=1.0
    )


class Severity(str, Enum):
    """Severity levels for security findings."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SecurityFinding(BaseModel):
    """A specific security issue found in the code."""

    model_config = ConfigDict(frozen=True)

    severity: Severity = Field(description="Severity of the finding")
    finding_type: str = Field(description="Type or category of the security issue")
    file_path: str = Field(description="Path to the file where the issue was found")
    line_number: int | None = Field(default=None, description="Line number of the issue")
    description: str = Field(description="Description of the security issue")
    recommendation: str = Field(description="Recommendation for fixing the issue")


class SecurityReport(BaseModel):
    """Overall security report."""

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(
        description="Whether the code passed security checks without critical/high findings"
    )
    findings: list[SecurityFinding] = Field(
        default_factory=list, description="List of security findings"
    )


class TokenUsage(BaseModel):
    """Tracking of token usage for a single API call."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(description="Model used for the generation")
    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")
    estimated_cost_usd: float = Field(description="Estimated cost of the call in USD")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the API call"
    )


class AgentTrace(BaseModel):
    """A trace entry representing a step in the agent's execution."""

    model_config = ConfigDict(frozen=True)

    step_name: str = Field(description="Name of the execution step")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the step"
    )
    duration_seconds: float = Field(description="Duration of the step in seconds")
    input_summary: str = Field(description="Summary of inputs provided to this step")
    output_summary: str = Field(description="Summary of outputs produced by this step")
    model_used: str | None = Field(
        default=None, description="Model used in this step, if applicable"
    )
    token_usage: TokenUsage | None = Field(
        default=None, description="Token usage for this step, if applicable"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the step"
    )


class AgentResult(BaseModel):
    """Final result produced by the agent."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(description="Whether the overall task was successful")
    plan: Plan | None = Field(default=None, description="The plan generated and followed")
    changes: list[CodeChange] = Field(default_factory=list, description="Code changes made")
    test_results: list[TestResult] = Field(
        default_factory=list, description="Results of tests run during execution"
    )
    security_report: SecurityReport | None = Field(
        default=None, description="Security scan results"
    )
    diff_text: str = Field(default="", description="The final diff of all changes")
    report_markdown: str = Field(default="", description="Markdown report of the agent's work")
    total_token_usage: list[TokenUsage] = Field(
        default_factory=list, description="All token usage entries"
    )
    traces: list[AgentTrace] = Field(default_factory=list, description="All trace entries")
    error_message: str | None = Field(
        default=None, description="Global error message if the task failed entirely"
    )


class RepoInfo(BaseModel):
    """Information about the repository being worked on."""

    model_config = ConfigDict(frozen=True)

    clone_path: str = Field(description="Local path where the repo is cloned")
    repo_url: str = Field(description="URL of the remote repository")
    default_branch: str = Field(description="Default branch name")
    file_tree: list[str] = Field(description="List of files in the repository")
    language_stats: dict[str, int] = Field(description="Mapping of language to bytes/lines of code")
