from __future__ import annotations

import threading

from patchpilot.llm.token_tracker import TokenTracker
from patchpilot.models import TokenUsage


def test_record_and_summary() -> None:
    """Record usage, verify summary totals."""
    tracker = TokenTracker()
    tracker.record(
        TokenUsage(
            model="claude-sonnet-4-20250514",
            input_tokens=10,
            output_tokens=20,
            estimated_cost_usd=0.001,
        )
    )
    tracker.record(
        TokenUsage(
            model="claude-sonnet-4-20250514",
            input_tokens=5,
            output_tokens=5,
            estimated_cost_usd=0.0005,
        )
    )
    summary = tracker.get_summary()
    assert summary["total_input_tokens"] == 15
    assert summary["total_output_tokens"] == 25
    assert summary["models"]["claude-sonnet-4-20250514"]["input_tokens"] == 15


def test_get_usage_by_model() -> None:
    """Filter by model name."""
    tracker = TokenTracker()
    tracker.record(
        TokenUsage(
            model="claude-sonnet-4-20250514",
            input_tokens=10,
            output_tokens=20,
            estimated_cost_usd=0.001,
        )
    )
    usage = tracker.get_usage_by_model("claude-sonnet-4-20250514")
    assert len(usage) == 1
    assert usage[0].input_tokens == 10


def test_reset_clears_records() -> None:
    """Reset empties the tracker."""
    tracker = TokenTracker()
    tracker.record(
        TokenUsage(
            model="claude-sonnet-4-20250514",
            input_tokens=10,
            output_tokens=20,
            estimated_cost_usd=0.001,
        )
    )
    tracker.reset()
    assert len(tracker.get_all_usage()) == 0


def test_to_report_markdown() -> None:
    """Verify report contains model name and formatted numbers."""
    tracker = TokenTracker()
    tracker.record(
        TokenUsage(
            model="claude-sonnet-4-20250514",
            input_tokens=10,
            output_tokens=20,
            estimated_cost_usd=0.001,
        )
    )
    report = tracker.to_report()
    assert "claude-sonnet-4-20250514" in report
    assert "10" in report


def test_thread_safety() -> None:
    """Record from multiple threads, verify all recorded."""
    tracker = TokenTracker()

    def worker() -> None:
        for _ in range(100):
            tracker.record(
                TokenUsage(
                    model="claude-sonnet-4-20250514",
                    input_tokens=1,
                    output_tokens=1,
                    estimated_cost_usd=0.0001,
                )
            )

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(tracker.get_all_usage()) == 1000


def test_get_all_usage() -> None:
    """Verify returns list copy of all records."""
    tracker = TokenTracker()
    tracker.record(
        TokenUsage(
            model="claude-sonnet-4-20250514",
            input_tokens=10,
            output_tokens=20,
            estimated_cost_usd=0.001,
        )
    )
    usage = tracker.get_all_usage()
    assert len(usage) == 1
    usage.clear()
    assert len(tracker.get_all_usage()) == 1
