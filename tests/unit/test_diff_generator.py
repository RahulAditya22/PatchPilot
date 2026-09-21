from __future__ import annotations

from patchpilot.models import CodeChange
from patchpilot.tools.diff_generator import DiffGenerator


def test_generate_unified_diff() -> None:
    """Verify diff output contains --- and +++ headers."""
    generator = DiffGenerator()
    change = CodeChange(
        file_path="test.py",
        original_content="line 1\n",
        modified_content="line 1\nline 2\n",
        change_description="Added line 2",
    )
    diff = generator.generate_unified_diff([change])
    assert "--- a/test.py" in diff
    assert "+++ b/test.py" in diff


def test_generate_diff_shows_changes() -> None:
    """Verify added/removed lines appear."""
    generator = DiffGenerator()
    change = CodeChange(
        file_path="test.py",
        original_content="old_line\n",
        modified_content="new_line\n",
        change_description="Changed line",
    )
    diff = generator.generate_unified_diff([change])
    assert "-old_line" in diff
    assert "+new_line" in diff


def test_empty_changes_returns_empty() -> None:
    """No changes = empty diff."""
    generator = DiffGenerator()
    diff = generator.generate_unified_diff([])
    assert diff == ""


def test_generate_pr_diff() -> None:
    """Verify PR-style formatting with git headers."""
    generator = DiffGenerator()
    change = CodeChange(
        file_path="test.py",
        original_content="",
        modified_content="new file content\n",
        change_description="Created test.py",
    )
    pr_diff = generator.generate_pr_diff([change])
    assert "diff --git a/test.py b/test.py" in pr_diff
    assert "new file mode 100644" in pr_diff
