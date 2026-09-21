from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from patchpilot.tools.indexer import CodebaseIndexer


def test_chunk_python_file() -> None:
    """Verify Python file is chunked by functions/classes."""
    with patch("patchpilot.tools.indexer.SentenceTransformerEmbeddingFunction"):
        with patch("chromadb.Client"):
            indexer = CodebaseIndexer()
            chunks = indexer._chunk_python_file("def a(): pass\nclass B: pass\n", "foo.py")
            assert len(chunks) == 2


def test_chunk_generic_file() -> None:
    """Verify line-based chunking with overlap."""
    with patch("patchpilot.tools.indexer.SentenceTransformerEmbeddingFunction"):
        with patch("chromadb.Client"):
            indexer = CodebaseIndexer()
            text = "line\n" * 100
            chunks = indexer._chunk_generic_file(text, "bar.txt")
            assert len(chunks) > 1


@pytest.mark.asyncio
async def test_index_repository(tmp_path: Path) -> None:
    """Mock chromadb, verify repository indexing."""
    f = tmp_path / "test.py"
    f.write_text("def foo(): return 1\n", encoding="utf-8")

    with patch("patchpilot.tools.indexer.SentenceTransformerEmbeddingFunction"):
        with patch("chromadb.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_cls.return_value = mock_client

            indexer = CodebaseIndexer()
            count = await indexer.index_repository(str(tmp_path))
            assert count == 1
            assert mock_collection.add.called
