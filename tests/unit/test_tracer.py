from __future__ import annotations

import pytest

from patchpilot.observability.tracer import AgentTracer


def test_agent_tracer_flow(tmp_path: pytest.TempPathFactory) -> None:
    tracer = AgentTracer(trace_dir=str(tmp_path))
    step_id = tracer.start_step("test_step", input_summary="input")
    trace = tracer.end_step(step_id, output_summary="output", model_used="claude")
    assert trace.step_name == "test_step"
    assert trace.output_summary == "output"

    traces = tracer.get_traces()
    assert len(traces) == 1

    saved_path = tracer.save("test_trace.json")
    assert saved_path.exists()

    report = tracer.to_markdown_report()
    assert "test_step" in report


def test_agent_tracer_context_manager() -> None:
    tracer = AgentTracer()
    with tracer.trace_step("cm_step", input_summary="in") as ctx:
        ctx["output_summary"] = "out"

    traces = tracer.get_traces()
    assert any(t.step_name == "cm_step" for t in traces)
