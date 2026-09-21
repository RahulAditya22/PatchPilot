from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from patchpilot.api.app import app
from patchpilot.models import AgentResult, Plan, RepoInfo, SecurityReport


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_run_endpoint_success(client: TestClient) -> None:
    with patch("patchpilot.api.routes.Orchestrator") as mock_orch_cls:
        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(
            return_value=AgentResult(
                success=True,
                plan=Plan(issue_summary="summary", steps=[], reasoning=""),
                changes=[],
                test_results=[],
                security_report=SecurityReport(passed=True, findings=[]),
                diff_text="diff",
                report_markdown="report",
                total_token_usage=[],
                traces=[],
                error_message=None,
            )
        )
        mock_orch_cls.return_value = mock_orch

        response = client.post(
            "/api/v1/run",
            json={
                "repo_url": "https://github.com/test/repo",
                "issue_text": "Fix a bug",
                "issue_title": "Bug",
                "max_retries": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


def test_analyze_endpoint_success(client: TestClient) -> None:
    with patch("patchpilot.api.routes.RepoIngester") as mock_ingester_cls:
        mock_ingester = AsyncMock()
        mock_ingester.ingest = AsyncMock(
            return_value=RepoInfo(
                clone_path="/tmp/repo",
                repo_url="https://github.com/test/repo",
                default_branch="main",
                file_tree=["a.py", "b.py"],
                language_stats={"Python": 2},
            )
        )
        mock_ingester_cls.return_value = mock_ingester

        response = client.post(
            "/api/v1/analyze",
            json={"repo_url": "https://github.com/test/repo"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["file_count"] == 2
        assert data["languages"]["Python"] == 2
