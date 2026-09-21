"""Code search engine — combined semantic + lexical search.

Model Attribution: Gemini (tool wiring)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from patchpilot.tools.indexer import CodebaseIndexer


class CodeSearchEngine:
    """Combined semantic + lexical search engine."""

    def __init__(self, indexer: CodebaseIndexer | None = None) -> None:
        """Initialize CodeSearchEngine.

        Args:
            indexer: CodebaseIndexer instance for semantic search.
        """
        self.indexer = indexer

    async def search(
        self,
        query: str,
        search_type: str = "hybrid",
        max_results: int = 10,
        repo_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search codebase using semantic, lexical, or hybrid approach.

        Args:
            query: The search query.
            search_type: 'semantic', 'lexical', or 'hybrid'.
            max_results: Maximum results to return.
            repo_path: Local path to repository.

        Returns:
            list[dict[str, Any]]: Search results.
        """
        results: list[dict[str, Any]] = []

        if search_type in ("semantic", "hybrid") and self.indexer:
            semantic_results = await self.indexer.search(query, n_results=max_results)
            for res in semantic_results:
                res["search_type"] = "semantic"
                results.append(res)

        if search_type in ("lexical", "hybrid") and repo_path:
            lexical_results = self._lexical_search(query, repo_path, max_results=max_results)
            for res in lexical_results:
                res["search_type"] = "lexical"
                results.append(res)

        if search_type == "hybrid":
            # Deduplicate by file_path and content, rank by score
            seen: set[tuple[str, int]] = set()
            hybrid_results: list[dict[str, Any]] = []

            # Simple scoring: sum scores from semantic and lexical
            file_scores: dict[str, float] = {}
            for res in results:
                fp = str(res["file_path"])
                file_scores[fp] = file_scores.get(fp, 0.0) + float(res.get("score", 0.5))

            for res in sorted(
                results,
                key=lambda x: float(file_scores.get(str(x["file_path"]), 0.0)),
                reverse=True,
            ):
                key = (str(res["file_path"]), int(res.get("start_line", 0)))
                if key not in seen:
                    seen.add(key)
                    hybrid_results.append(res)

            results = hybrid_results[:max_results]

        return results

    def _lexical_search(
        self, query: str, repo_path: str, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Simple grep-like search through files.

        Args:
            query: The search query.
            repo_path: Path to repository.
            max_results: Max results.

        Returns:
            list[dict[str, Any]]: Lexical search results.
        """
        repo_dir = Path(repo_path)
        results: list[dict[str, Any]] = []
        query_lower = query.lower()

        for root, dirs, files in os.walk(repo_dir):
            if ".git" in dirs:
                dirs.remove(".git")

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in {".png", ".jpg", ".pyc", ".exe", ".dll", ".so"}:
                    continue

                try:
                    with open(file_path, encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                except Exception:
                    continue

                match_count = 0
                matching_lines: list[int] = []

                for i, line in enumerate(lines):
                    if query_lower in line.lower():
                        match_count += 1
                        matching_lines.append(i)

                if match_count > 0:
                    content = "".join(lines)
                    rel_path = file_path.relative_to(repo_dir).as_posix()

                    results.append(
                        {
                            "file_path": rel_path,
                            "content": content,
                            "score": min(1.0, match_count * 0.1),
                            "match_count": match_count,
                        }
                    )

        return sorted(results, key=lambda x: float(x["score"]), reverse=True)[:max_results]
