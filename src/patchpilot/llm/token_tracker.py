"""Token and cost tracking for LLM API calls.

Thread-safe accumulation of per-call TokenUsage records with
summary reports broken down by model.

Model Attribution: Gemini (utility wiring)
"""

from __future__ import annotations

import threading
from typing import Any

from patchpilot.models import TokenUsage


class TokenTracker:
    """Thread-safe tracker for LLM token usage and costs."""

    def __init__(self) -> None:
        """Initialize the token tracker."""
        self._lock = threading.Lock()
        self._records: list[TokenUsage] = []

    def record(self, usage: TokenUsage) -> None:
        """Record a new token usage entry.

        Args:
            usage: The TokenUsage record to add.
        """
        with self._lock:
            self._records.append(usage)

    def get_all_usage(self) -> list[TokenUsage]:
        """Return a copy of all usage records.

        Returns:
            List of all TokenUsage records.
        """
        with self._lock:
            return list(self._records)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all token usage and costs.

        Returns:
            Dictionary with total tokens, total cost, and breakdown by model.
        """
        with self._lock:
            total_input = 0
            total_output = 0
            total_cost = 0.0
            models_breakdown: dict[str, dict[str, Any]] = {}

            for r in self._records:
                total_input += r.input_tokens
                total_output += r.output_tokens
                total_cost += r.estimated_cost_usd

                model = r.model
                if model not in models_breakdown:
                    models_breakdown[model] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "call_count": 0,
                    }

                models_breakdown[model]["input_tokens"] += r.input_tokens
                models_breakdown[model]["output_tokens"] += r.output_tokens
                models_breakdown[model]["total_tokens"] += r.input_tokens + r.output_tokens
                models_breakdown[model]["estimated_cost_usd"] += r.estimated_cost_usd
                models_breakdown[model]["call_count"] += 1

            return {
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "total_cost_usd": total_cost,
                "call_count": len(self._records),
                "models": models_breakdown,
            }

    def get_usage_by_model(self, model: str) -> list[TokenUsage]:
        """Get all usage records for a specific model.

        Args:
            model: The model name to filter by.

        Returns:
            List of TokenUsage records for the specified model.
        """
        with self._lock:
            return [r for r in self._records if r.model == model]

    def reset(self) -> None:
        """Clear all usage records."""
        with self._lock:
            self._records.clear()

    def to_report(self) -> str:
        """Generate a formatted markdown report of all usage.

        Returns:
            Markdown string with usage summary table.
        """
        summary = self.get_summary()

        report_lines = [
            "# LLM Token Usage Report",
            "",
            "## Overall Summary",
            f"- **Total Input Tokens:** {summary['total_input_tokens']:,}",
            f"- **Total Output Tokens:** {summary['total_output_tokens']:,}",
            f"- **Total Tokens:** {summary['total_tokens']:,}",
            f"- **Total API Calls:** {summary['call_count']}",
            f"- **Total Estimated Cost:** \\${summary['total_cost_usd']:.4f}",
            "",
            "## Breakdown by Model",
            "",
        ]

        if not summary["models"]:
            report_lines.append("No usage recorded yet.")
        else:
            report_lines.append(
                "| Model | Calls | Input Tokens | Output Tokens | Total Tokens | Cost |"
            )
            report_lines.append(
                "|-------|-------|--------------|---------------|--------------|------|"
            )
            for model_name, data in summary["models"].items():
                report_lines.append(
                    f"| {model_name} | {data['call_count']} | "
                    f"{data['input_tokens']:,} | {data['output_tokens']:,} | "
                    f"{data['total_tokens']:,} | \\${data['estimated_cost_usd']:.4f} |"
                )

        return "\n".join(report_lines)


# Singleton instance
_tracker: TokenTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> TokenTracker:
    """Get the module-level singleton TokenTracker instance.

    Returns:
        The shared TokenTracker instance.
    """
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = TokenTracker()
    return _tracker
