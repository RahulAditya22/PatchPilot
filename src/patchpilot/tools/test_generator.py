"""Test generator — produces test skeletons for code changes.

Model Attribution: Gemini (tool wiring)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from patchpilot.models import CodeChange


class TestGenerator:
    """Generates tests for code changes."""

    def __init__(self, llm_client: Any | None = None) -> None:
        """Initialize TestGenerator.

        Args:
            llm_client: Optional LLM client to use for smart generation.
        """
        self.llm_client = llm_client

    async def generate_tests(self, changes: list[CodeChange], repo_path: str) -> list[CodeChange]:
        """Generate test files for the given changes.

        Args:
            changes: List of code changes.
            repo_path: Local repository path.

        Returns:
            list[CodeChange]: List of test file code changes.
        """
        test_changes = []

        for change in changes:
            if not change.file_path.endswith(".py") or "test_" in change.file_path:
                continue

            test_change = self._generate_skeleton_test(change)
            if test_change:
                test_changes.append(test_change)

        return test_changes

    def _generate_skeleton_test(self, change: CodeChange) -> CodeChange | None:
        """Generate a basic pytest test skeleton for a changed file.

        Args:
            change: The original code change.

        Returns:
            CodeChange: The test code change, or None if it couldn't be generated.
        """
        if not change.modified_content:
            return None

        try:
            tree = ast.parse(change.modified_content)
        except SyntaxError:
            return None

        test_file_path = f"tests/test_{Path(change.file_path).name}"
        test_content = ["import pytest", f"from {Path(change.file_path).stem} import *", ""]

        has_tests = False

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                test_content.append(f"def test_{node.name}():")
                test_content.append(f"    # TODO: Implement test for {node.name}")
                test_content.append("    pass\n")
                has_tests = True
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                test_content.append(f"class Test{node.name}:")
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                        test_content.append(f"    def test_{item.name}(self):")
                        test_content.append(
                            f"        # TODO: Implement test for {node.name}.{item.name}"
                        )
                        test_content.append("        pass\n")
                        has_tests = True

        if not has_tests:
            return None

        return CodeChange(
            file_path=test_file_path,
            original_content="",
            modified_content="\n".join(test_content),
            change_description=f"Generated test skeleton for {change.file_path}",
        )
