"""Evaluation benchmark runner for PatchPilot agent performance.

Evaluates PatchPilot across standard test scenarios, measuring success rate,
execution time, token usage, and cost per model.

Model Attribution: Claude Opus (evaluation suite design)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from patchpilot.agents.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runner for evaluating agent performance on scenarios."""

    def __init__(self, scenarios: list[dict[str, Any]] | None = None) -> None:
        """Initialize the EvaluationRunner.

        Args:
            scenarios: List of scenario dicts with repo_url, issue_text, title.
        """
        self.scenarios = scenarios or [
            {
                "id": "scenario_1_bugfix",
                "name": "Simple Bugfix",
                "repo_url": "https://github.com/test/sample_repo_1",
                "issue_text": "Fix AttributeError in string parsing",
                "issue_title": "Fix string parsing crash",
            },
            {
                "id": "scenario_2_feature",
                "name": "Feature Addition",
                "repo_url": "https://github.com/test/sample_repo_2",
                "issue_text": "Add JSON export capability to exporter module",
                "issue_title": "JSON export feature",
            },
            {
                "id": "scenario_3_adversarial",
                "name": "Adversarial Edge Case",
                "repo_url": "https://github.com/test/sample_repo_3",
                "issue_text": "Malformed issue with ambiguous requirements and circular logic",
                "issue_title": "Edge case issue",
            },
        ]
        self.results: list[dict[str, Any]] = []

    async def run_all(self) -> dict[str, Any]:
        """Run all evaluation scenarios and compute aggregate metrics.

        Returns:
            Dictionary containing success_rate, avg_time, avg_cost, total_runs.
        """
        orchestrator = Orchestrator()
        self.results.clear()

        for scenario in self.scenarios:
            logger.info("Running evaluation scenario: %s", scenario["name"])
            start_time = time.monotonic()

            try:
                result = await orchestrator.run(
                    repo_url=scenario["repo_url"],
                    issue_text=scenario["issue_text"],
                    issue_title=scenario.get("issue_title", ""),
                )
                duration = time.monotonic() - start_time

                # Extract token usage and cost
                total_cost = sum(t.estimated_cost_usd for t in result.total_token_usage)
                total_tokens = sum(
                    t.input_tokens + t.output_tokens for t in result.total_token_usage
                )

                self.results.append(
                    {
                        "id": scenario["id"],
                        "name": scenario["name"],
                        "success": result.success,
                        "duration_seconds": duration,
                        "total_tokens": total_tokens,
                        "total_cost_usd": total_cost,
                        "error_message": result.error_message,
                    }
                )
            except Exception as exc:
                duration = time.monotonic() - start_time
                self.results.append(
                    {
                        "id": scenario["id"],
                        "name": scenario["name"],
                        "success": False,
                        "duration_seconds": duration,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0,
                        "error_message": str(exc),
                    }
                )

        return self.compute_metrics()

    def compute_metrics(self) -> dict[str, Any]:
        """Compute summary metrics from current results.

        Returns:
            Dict of aggregate metrics.
        """
        if not self.results:
            return {
                "success_rate": 0.0,
                "avg_time_seconds": 0.0,
                "avg_cost_usd": 0.0,
                "total_runs": 0,
                "successful_runs": 0,
            }

        total_runs = len(self.results)
        successful_runs = sum(1 for r in self.results if r["success"])
        success_rate = successful_runs / total_runs
        avg_time = sum(r["duration_seconds"] for r in self.results) / total_runs
        avg_cost = sum(r["total_cost_usd"] for r in self.results) / total_runs

        return {
            "success_rate": success_rate,
            "avg_time_seconds": avg_time,
            "avg_cost_usd": avg_cost,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
        }

    def generate_report(self) -> str:
        """Generate markdown evaluation report.

        Returns:
            Formatted markdown report.
        """
        metrics = self.compute_metrics()

        lines = [
            "# PatchPilot Evaluation Benchmark Report",
            "",
            "## Aggregate Metrics",
            f"- **Success Rate:** {metrics['success_rate'] * 100:.1f}% ({metrics['successful_runs']}/{metrics['total_runs']})",
            f"- **Average Time per Run:** {metrics['avg_time_seconds']:.2f}s",
            f"- **Average Cost per Run:** \\${metrics['avg_cost_usd']:.4f}",
            "",
            "## Scenario Details",
            "",
            "| Scenario | Status | Time (s) | Total Tokens | Cost (USD) | Error |",
            "|----------|--------|----------|--------------|------------|-------|",
        ]

        for r in self.results:
            status = "✅ Pass" if r["success"] else "❌ Fail"
            err = r.get("error_message") or "-"
            lines.append(
                f"| {r['name']} | {status} | {r['duration_seconds']:.1f} | "
                f"{r['total_tokens']:,} | \\${r['total_cost_usd']:.4f} | {err} |"
            )

        if not self.results:
            lines.append("*(No scenarios run yet)*")

        return "\n".join(lines)
