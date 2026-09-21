from __future__ import annotations

from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from patchpilot.cli.main import cli
from patchpilot.models import AgentResult, Plan, RepoInfo, SecurityReport


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "PatchPilot" in result.output


def test_cli_run_command() -> None:
    runner = CliRunner()
    with patch("patchpilot.cli.main.Orchestrator") as mock_orch_cls:
        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(
            return_value=AgentResult(
                success=True,
                plan=Plan(issue_summary="summary", steps=[], reasoning=""),
                changes=[],
                test_results=[],
                security_report=SecurityReport(passed=True, findings=[]),
                diff_text="diff text",
                report_markdown="report text",
                total_token_usage=[],
                traces=[],
                error_message=None,
            )
        )
        mock_orch_cls.return_value = mock_orch

        result = runner.invoke(
            cli, ["run", "--repo", "https://github.com/test/repo", "--issue", "fix bug"]
        )
        assert result.exit_code == 0
        assert "Run completed" in result.output


def test_cli_analyze_command() -> None:
    runner = CliRunner()
    with patch(
        "patchpilot.tools.repo_ingester.RepoIngester.ingest", new_callable=AsyncMock
    ) as mock_ingest:
        mock_ingest.return_value = RepoInfo(
            clone_path="/tmp/repo",
            repo_url="https://github.com/test/repo",
            default_branch="main",
            file_tree=["a.py"],
            language_stats={"Python": 1},
        )
        result = runner.invoke(cli, ["analyze", "--repo", "https://github.com/test/repo"])
        assert result.exit_code == 0
        assert "Files indexed: 1" in result.output
