from __future__ import annotations

import pytest

from patchpilot.models import CodeChange
from patchpilot.tools.test_generator import TestGenerator


@pytest.mark.asyncio
async def test_generate_skeleton_tests() -> None:
    generator = TestGenerator()
    change = CodeChange(
        file_path="src/math_utils.py",
        original_content="",
        modified_content="def add(a, b):\n    return a + b\n\nclass Calculator:\n    def subtract(self, a, b):\n        return a - b\n",
        change_description="added math utils",
    )

    test_changes = await generator.generate_tests([change], repo_path="/tmp")
    assert len(test_changes) == 1
    tc = test_changes[0]
    assert "test_math_utils.py" in tc.file_path
    assert "def test_add" in tc.modified_content
    assert "class TestCalculator" in tc.modified_content
