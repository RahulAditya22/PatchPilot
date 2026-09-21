from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from patchpilot.tools.code_search import CodeSearchEngine


@pytest.mark.asyncio
async def test_code_search_hybrid(tmp_path: Path) -> None:
    f = tmp_path / "main.py"
    f.write_text("def parse_data(): pass\n", encoding="utf-8")

    mock_indexer = AsyncMock()
    mock_indexer.search = AsyncMock(
        return_value=[
            {
                "file_path": "main.py",
                "content": "def parse_data(): pass",
                "score": 0.9,
                "start_line": 1,
                "end_line": 1,
            }
        ]
    )

    engine = CodeSearchEngine(indexer=mock_indexer)
    results = await engine.search(
        query="parse",
        search_type="hybrid",
        max_results=5,
        repo_path=str(tmp_path),
    )
    assert len(results) >= 1
    assert results[0]["file_path"] == "main.py"
