from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from patchpilot.tools.repo_ingester import RepoIngester


@pytest.mark.asyncio
async def test_ingest_creates_repo_info(tmp_path: Path) -> None:
    """Mock git.Repo.clone_from, verify RepoInfo returned."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    with patch("git.Repo.clone_from") as mock_clone:
        mock_repo = MagicMock()
        mock_repo.active_branch = "main"
        mock_clone.return_value = mock_repo

        ingester = RepoIngester()
        info = await ingester.ingest("https://github.com/test/repo", clone_dir=str(tmp_path))
        assert info.repo_url == "https://github.com/test/repo"
        assert info.default_branch == "main"
        assert "src/main.py" in info.file_tree
        assert "image.png" not in info.file_tree
        assert info.language_stats.get("Python", 0) == 1
