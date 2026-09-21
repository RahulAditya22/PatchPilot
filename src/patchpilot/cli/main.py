"""Command-line interface for PatchPilot.

Model Attribution: Gemini (CLI wiring)
"""

from __future__ import annotations

import asyncio
import os

import click
import uvicorn
from rich.console import Console
from rich.panel import Panel

from patchpilot.agents.orchestrator import Orchestrator

console = Console()


@click.group()
def cli() -> None:
    """PatchPilot — AI Software Engineering Agent CLI."""
    pass


@cli.command()
@click.option("--repo", "-r", required=True, help="GitHub repository URL")
@click.option("--issue", "-i", required=True, help="Issue text or GitHub issue URL")
@click.option("--title", "-t", default="", help="Issue title")
@click.option("--max-retries", default=3, help="Max retry attempts")
@click.option("--output", "-o", default=None, help="Output directory for results")
def run(repo: str, issue: str, title: str, max_retries: int, output: str | None) -> None:
    """Run PatchPilot on a repository with an issue.

    Args:
        repo: GitHub repository URL or local path.
        issue: Issue description text.
        title: Optional issue title.
        max_retries: Max retries for test failures.
        output: Optional directory to save diff and report.
    """

    async def _run() -> None:
        console.print(
            Panel(f"Starting PatchPilot run for [bold blue]{repo}[/bold blue]", title="PatchPilot")
        )
        try:
            orchestrator = Orchestrator()
            result = await orchestrator.run(
                repo_url=repo, issue_text=issue, issue_title=title, max_retries=max_retries
            )

            status_color = "green" if result.success else "red"
            console.print(
                f"[{status_color}]Run completed. Success: {result.success}[/{status_color}]"
            )

            if output:
                os.makedirs(output, exist_ok=True)
                report_path = os.path.join(output, "report.md")
                diff_path = os.path.join(output, "changes.diff")

                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(result.report_markdown)

                if result.diff_text:
                    with open(diff_path, "w", encoding="utf-8") as f:
                        f.write(result.diff_text)

                console.print(f"Results saved to [bold cyan]{output}[/bold cyan]")
            else:
                console.print(result.report_markdown)

        except Exception as e:
            console.print(f"[bold red]Error during run: {e}[/bold red]")

    asyncio.run(_run())


@cli.command()
@click.option("--repo", "-r", required=True, help="GitHub repository URL")
def analyze(repo: str) -> None:
    """Analyze a repository structure.

    Args:
        repo: GitHub repository URL or local path.
    """

    async def _analyze() -> None:
        console.print(f"Analyzing repository: [bold blue]{repo}[/bold blue]")
        try:
            from patchpilot.tools.repo_ingester import RepoIngester

            ingester = RepoIngester()
            info = await ingester.ingest(repo)
            console.print(f"[green]Files indexed:[/green] {len(info.file_tree)}")
            console.print(f"[green]Languages:[/green] {info.language_stats}")
        except Exception as e:
            console.print(f"[bold red]Error during analysis: {e}[/bold red]")

    asyncio.run(_analyze())


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def serve(host: str, port: int) -> None:
    """Start the API server.

    Args:
        host: Host to bind.
        port: Port to bind.
    """
    console.print(f"Starting PatchPilot API server on [bold green]{host}:{port}[/bold green]")
    uvicorn.run("patchpilot.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()
