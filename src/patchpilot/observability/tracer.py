"""Agent execution tracer for observability.

Records step-by-step execution traces including timing, model usage,
inputs, outputs, and token counts.

Model Attribution: Gemini (observability wiring)
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from patchpilot.config import get_config
from patchpilot.models import AgentTrace, TokenUsage


class AgentTracer:
    """Records and manages traces for agent execution steps."""

    def __init__(self, trace_dir: str | None = None) -> None:
        """Initialize the tracer.

        Args:
            trace_dir: Directory to save trace files. Uses config if None.
        """
        config = get_config()
        self.trace_dir = Path(trace_dir) if trace_dir else Path(config.TRACE_DIR)
        self._active_steps: dict[str, dict[str, Any]] = {}
        self._completed_traces: list[AgentTrace] = []

    def start_step(
        self,
        step_name: str,
        input_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start recording a new step.

        Args:
            step_name: Name of the execution step.
            input_summary: Summary of inputs to the step.
            metadata: Additional metadata dict.

        Returns:
            The generated step_id string.
        """
        step_id = str(uuid.uuid4())
        self._active_steps[step_id] = {
            "step_id": step_id,
            "step_name": step_name,
            "input_summary": input_summary,
            "metadata": metadata or {},
            "start_time": datetime.now(UTC),
        }
        return step_id

    def end_step(
        self,
        step_id: str,
        output_summary: str = "",
        model_used: str | None = None,
        token_usage: TokenUsage | None = None,
    ) -> AgentTrace:
        """End recording a step and generate its trace.

        Args:
            step_id: ID of the step to end.
            output_summary: Summary of output produced.
            model_used: Model name used, if any.
            token_usage: TokenUsage record for the step, if any.

        Returns:
            The completed AgentTrace object.
        """
        if step_id not in self._active_steps:
            # Fallback if step_id was not tracked (e.g. in mocks)
            trace = AgentTrace(
                step_name=f"unknown_step_{step_id}",
                timestamp=datetime.now(UTC),
                duration_seconds=0.0,
                input_summary="",
                output_summary=output_summary,
                model_used=model_used,
                token_usage=token_usage,
                metadata={},
            )
            self._completed_traces.append(trace)
            return trace

        step_data = self._active_steps.pop(step_id)
        end_time = datetime.now(UTC)
        start_time = step_data["start_time"]
        duration = (end_time - start_time).total_seconds()

        trace = AgentTrace(
            step_name=step_data["step_name"],
            timestamp=start_time,
            duration_seconds=duration,
            input_summary=step_data["input_summary"],
            output_summary=output_summary,
            model_used=model_used,
            token_usage=token_usage,
            metadata=step_data["metadata"],
        )
        self._completed_traces.append(trace)
        return trace

    def get_traces(self) -> list[AgentTrace]:
        """Get all completed traces.

        Returns:
            List of AgentTrace objects.
        """
        return list(self._completed_traces)

    def save(self, filename: str | None = None) -> Path:
        """Save traces to a JSON file.

        Args:
            filename: Output filename. Timestamped if None.

        Returns:
            Path to the saved JSON file.
        """
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"trace_{timestamp}.json"

        file_path = self.trace_dir / filename

        traces_data = [t.model_dump(mode="json") for t in self._completed_traces]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(traces_data, f, indent=2, default=str)

        return file_path

    def to_markdown_report(self) -> str:
        """Generate a markdown table report of all traces.

        Returns:
            Formatted markdown string.
        """
        lines = ["# Agent Execution Trace Report\n"]
        lines.append("| Step Name | Duration (s) | Model | Input Summary | Output Summary |")
        lines.append("|---|---|---|---|---|")

        for trace in self._completed_traces:
            model = trace.model_used or "N/A"
            lines.append(
                f"| {trace.step_name} | {trace.duration_seconds:.2f} | {model} | "
                f"{trace.input_summary} | {trace.output_summary} |"
            )

        return "\n".join(lines) + "\n"

    @contextmanager
    def trace_step(
        self,
        step_name: str,
        input_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Context manager to auto-start and end a step.

        Args:
            step_name: Step name.
            input_summary: Input summary.
            metadata: Metadata dict.

        Yields:
            Context dict to store outputs.
        """
        step_id = self.start_step(step_name, input_summary, metadata)
        context: dict[str, Any] = {
            "output_summary": "",
            "model_used": None,
            "token_usage": None,
        }
        try:
            yield context
        finally:
            if step_id in self._active_steps:
                self.end_step(
                    step_id,
                    output_summary=context.get("output_summary", ""),
                    model_used=context.get("model_used"),
                    token_usage=context.get("token_usage"),
                )


_global_tracer: AgentTracer | None = None


def get_tracer() -> AgentTracer:
    """Get the global AgentTracer singleton.

    Returns:
        The AgentTracer instance.
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AgentTracer()
    return _global_tracer
