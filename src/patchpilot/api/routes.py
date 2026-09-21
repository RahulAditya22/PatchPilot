"""FastAPI route definitions for PatchPilot API.

Model Attribution: Gemini (API routing)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from patchpilot.agents.orchestrator import Orchestrator
from patchpilot.models import AgentResult, RepoInfo
from patchpilot.tools.repo_ingester import RepoIngester

router = APIRouter(prefix="/api/v1", tags=["agent"])


class RunRequest(BaseModel):
    """Request model for starting an agent run."""

    model_config = ConfigDict(extra="forbid")

    repo_url: str
    issue_text: str
    issue_title: str = ""
    max_retries: int = 3


class HealthResponse(BaseModel):
    """Response model for health check."""

    model_config = ConfigDict(extra="forbid")

    status: str
    version: str


class AnalyzeRequest(BaseModel):
    """Request model for analyzing a repository."""

    model_config = ConfigDict(extra="forbid")

    repo_url: str


class AnalyzeResponse(BaseModel):
    """Response model for repository analysis."""

    model_config = ConfigDict(extra="forbid")

    repo_info: RepoInfo
    file_count: int
    languages: dict[str, int]


@router.post("/run", response_model=AgentResult)
async def run_agent(request: RunRequest) -> AgentResult:
    """Start an agent run on a repository with an issue.

    Args:
        request: The request payload containing repository and issue details.

    Returns:
        AgentResult: The result of the agent run.

    Raises:
        HTTPException: If an error occurs during the run.
    """
    try:
        orchestrator = Orchestrator()
        result = await orchestrator.run(
            repo_url=request.repo_url,
            issue_text=request.issue_text,
            issue_title=request.issue_title,
            max_retries=request.max_retries,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Perform a health check on the API.

    Returns:
        HealthResponse: The health status and version.
    """
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a repository without making changes.

    Args:
        request: The request payload containing the repository URL.

    Returns:
        AnalyzeResponse: Information about the repository.

    Raises:
        HTTPException: If an error occurs during analysis.
    """
    try:
        ingester = RepoIngester()
        repo_info = await ingester.ingest(request.repo_url)
        return AnalyzeResponse(
            repo_info=repo_info,
            file_count=len(repo_info.file_tree),
            languages=repo_info.language_stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
