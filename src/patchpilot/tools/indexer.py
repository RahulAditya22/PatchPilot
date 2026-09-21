"""Codebase indexer — chunks and embeds code files for ChromaDB vector search.

Model Attribution: Gemini (tool wiring)
"""

from __future__ import annotations

import ast
import os
import uuid
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError:
    chromadb = None  # type: ignore[assignment]
    SentenceTransformerEmbeddingFunction = None  # type: ignore[assignment, misc]

from patchpilot.config import get_config


class CodebaseIndexer:
    """Indexes codebase for semantic search."""

    def __init__(self, persist_dir: str | None = None, embedding_model: str | None = None) -> None:
        """Initialize the indexer.

        Args:
            persist_dir: Directory to persist ChromaDB index.
            embedding_model: Name of the sentence transformer model to use.
        """
        if chromadb is None:
            raise ImportError("chromadb is required for CodebaseIndexer")

        config = get_config()
        model_name = embedding_model or config.EMBEDDING_MODEL

        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)

        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="patchpilot_code",
            embedding_function=self.embedding_fn,  # type: ignore[arg-type]
        )

    async def index_repository(self, repo_path: str) -> int:
        """Index all code files, return count of chunks indexed.

        Args:
            repo_path: Path to the local repository.

        Returns:
            int: Number of chunks indexed.
        """
        repo_dir = Path(repo_path)
        chunks: list[dict[str, Any]] = []

        for root, dirs, files in os.walk(repo_dir):
            if ".git" in dirs:
                dirs.remove(".git")

            for file in files:
                file_path = Path(root) / file

                if file_path.suffix.lower() in {".png", ".jpg", ".pyc", ".exe", ".dll", ".so"}:
                    continue

                try:
                    with open(file_path, encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception:
                    continue

                rel_path = file_path.relative_to(repo_dir).as_posix()

                if file_path.suffix == ".py":
                    file_chunks = self._chunk_python_file(content, rel_path)
                else:
                    file_chunks = self._chunk_generic_file(content, rel_path)

                chunks.extend(file_chunks)

        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [str(c["content"]) for c in chunks]
        metadatas = [
            {
                "file_path": str(c["file_path"]),
                "start_line": int(c["start_line"]),
                "end_line": int(c["end_line"]),
                "type": str(c.get("type", "generic")),
            }
            for c in chunks
        ]

        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],  # type: ignore[arg-type]
            )

        return len(chunks)

    async def search(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """Semantic search against indexed code.

        Args:
            query: The search query.
            n_results: Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: List of search results.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        output: list[dict[str, Any]] = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = (
                results.get("metadatas", [[]])[0] if results.get("metadatas") else [{}] * len(docs)
            )  # type: ignore[index]
            distances = (
                results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(docs)
            )  # type: ignore[index]

            for doc, meta, dist in zip(docs, metas, distances):
                output.append(
                    {
                        "file_path": meta.get("file_path", ""),
                        "content": doc,
                        "score": 1.0 / (1.0 + float(dist)),
                        "start_line": meta.get("start_line", 0),
                        "end_line": meta.get("end_line", 0),
                        "type": meta.get("type", "generic"),
                    }
                )

        return output

    def _chunk_python_file(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """Chunk Python files by AST nodes (functions, classes).

        Args:
            content: File content.
            file_path: File path.

        Returns:
            list[dict[str, Any]]: List of chunk dicts.
        """
        chunks: list[dict[str, Any]] = []
        lines = content.splitlines(keepends=True)

        try:
            tree = ast.parse(content, filename=file_path)

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line

                    node_content = "".join(lines[start_line - 1 : end_line])
                    chunks.append(
                        {
                            "content": node_content,
                            "file_path": file_path,
                            "start_line": start_line,
                            "end_line": end_line,
                            "type": "class" if isinstance(node, ast.ClassDef) else "function",
                        }
                    )

            if not chunks:
                return self._chunk_generic_file(content, file_path)

        except SyntaxError:
            return self._chunk_generic_file(content, file_path)

        return chunks

    def _chunk_generic_file(
        self, content: str, file_path: str, chunk_size: int = 50
    ) -> list[dict[str, Any]]:
        """Chunk non-Python files by fixed line count with overlap.

        Args:
            content: File content.
            file_path: File path.
            chunk_size: Number of lines per chunk.

        Returns:
            list[dict[str, Any]]: List of chunk dicts.
        """
        chunks: list[dict[str, Any]] = []
        lines = content.splitlines(keepends=True)
        overlap = 10

        i = 0
        while i < len(lines):
            end = min(i + chunk_size, len(lines))
            chunk_content = "".join(lines[i:end])

            chunks.append(
                {
                    "content": chunk_content,
                    "file_path": file_path,
                    "start_line": i + 1,
                    "end_line": end,
                    "type": "generic",
                }
            )

            i += chunk_size - overlap

        return chunks
