from __future__ import annotations

import ast
import os
from pathlib import Path


class DependencyAnalyzer:
    """Analyzes file dependencies."""

    async def analyze(self, file_path: str) -> list[str]:
        """Return list of files/modules this file depends on.

        Args:
            file_path: Absolute path to the Python file.

        Returns:
            list[str]: List of imported module names/paths.
        """
        path = Path(file_path)
        if not path.exists() or path.suffix != ".py":
            return []

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            tree = ast.parse(content, filename=str(path))
        except (SyntaxError, Exception):
            return []

        dependencies = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle relative imports roughly
                    module_name = node.module
                    if node.level > 0:
                        module_name = "." * node.level + module_name
                    dependencies.add(module_name)

        return sorted(list(dependencies))

    async def build_dependency_graph(self, repo_path: str) -> dict[str, list[str]]:
        """Build full dependency graph for a repository.

        Args:
            repo_path: Local path to the repository.

        Returns:
            dict[str, list[str]]: Map of file paths to their dependencies.
        """
        graph: dict[str, list[str]] = {}
        repo_dir = Path(repo_path)

        for root, dirs, files in os.walk(repo_dir):
            if ".git" in dirs:
                dirs.remove(".git")

            for file in files:
                if file.endswith(".py"):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(repo_dir).as_posix()
                    deps = await self.analyze(str(full_path))
                    graph[rel_path] = deps

        return graph
