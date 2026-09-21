"""Diff generator for PatchPilot.

Generates unified diffs, patch files, and PR-style diffs from CodeChange objects.

Model Attribution: Gemini (tool wiring)
"""

from __future__ import annotations

import difflib

from patchpilot.models import CodeChange


class DiffGenerator:
    """Generates clean diffs from code changes."""

    def generate_unified_diff(self, changes: list[CodeChange]) -> str:
        """Generate a unified diff from all changes.

        Args:
            changes: List of code changes.

        Returns:
            Unified diff text string.
        """
        diff_lines: list[str] = []
        for change in changes:
            old_lines = (
                change.original_content.splitlines(keepends=True) if change.original_content else []
            )
            new_lines = (
                change.modified_content.splitlines(keepends=True) if change.modified_content else []
            )

            file_diff = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{change.file_path}",
                tofile=f"b/{change.file_path}",
                n=3,
            )
            diff_lines.extend(list(file_diff))

        return "".join(diff_lines)

    def generate_patch_file(self, changes: list[CodeChange], output_path: str) -> str:
        """Write a .patch file.

        Args:
            changes: List of code changes.
            output_path: Path to write the patch file.

        Returns:
            Path to the generated patch file.
        """
        diff_text = self.generate_unified_diff(changes)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(diff_text)

        return output_path

    def generate_pr_diff(self, changes: list[CodeChange]) -> str:
        """Generate GitHub PR-style diff with file headers.

        Args:
            changes: List of code changes.

        Returns:
            PR-style diff text.
        """
        diff_text: list[str] = []

        for change in changes:
            diff_text.append(f"diff --git a/{change.file_path} b/{change.file_path}\n")
            if not change.original_content:
                diff_text.append("new file mode 100644\n")
            elif not change.modified_content:
                diff_text.append("deleted file mode 100644\n")

            old_lines = (
                change.original_content.splitlines(keepends=True) if change.original_content else []
            )
            new_lines = (
                change.modified_content.splitlines(keepends=True) if change.modified_content else []
            )

            file_diff = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{change.file_path}",
                tofile=f"b/{change.file_path}",
                n=3,
            )
            diff_text.extend(list(file_diff))

        return "".join(diff_text)
