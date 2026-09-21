from __future__ import annotations

import pytest

from patchpilot.tools.issue_parser import IssueParser


@pytest.mark.asyncio
async def test_parse_bullet_points() -> None:
    """Parse body with '- item1\n- item2', verify requirements."""
    parser = IssueParser()
    issue = await parser.parse("Fix auth", "- fix login\n- fix logout")
    assert len(issue.requirements) == 2
    assert "fix login" in issue.requirements[0]
    assert "fix logout" in issue.requirements[1]


@pytest.mark.asyncio
async def test_parse_numbered_list() -> None:
    """Parse '1. item1\n2. item2'."""
    parser = IssueParser()
    issue = await parser.parse("Fix auth", "1. fix login\n2. fix logout")
    assert len(issue.requirements) == 2
    assert "fix login" in issue.requirements[0]


@pytest.mark.asyncio
async def test_parse_priority_from_labels() -> None:
    """labels=['critical'] -> priority='critical'."""
    parser = IssueParser()
    issue = await parser.parse("Fix auth", "Fix issue immediately", labels=["critical", "bug"])
    assert issue.priority == "critical"


@pytest.mark.asyncio
async def test_parse_empty_body() -> None:
    """Empty body returns empty requirements list."""
    parser = IssueParser()
    issue = await parser.parse("Fix auth", "")
    assert len(issue.requirements) == 0


@pytest.mark.asyncio
async def test_parse_with_code_blocks() -> None:
    """Body with code blocks extracts requirements correctly."""
    parser = IssueParser()
    issue = await parser.parse("Fix code", "- fix this code:\n```python\nx=1\n```")
    assert len(issue.requirements) >= 1
    assert "fix this code" in issue.requirements[0]
