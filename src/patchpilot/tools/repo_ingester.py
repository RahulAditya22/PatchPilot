"""Repository ingester — clones GitHub repos and parses structure.

Model Attribution: Gemini (tool wiring)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import git

from patchpilot.models import RepoInfo


class RepoIngester:
    """Clones and parses GitHub repositories."""

    async def ingest(
        self,
        repo_url: str,
        clone_dir: str | None = None,
    ) -> RepoInfo:
        """Clone or update a repo and parse its structure.

        Args:
            repo_url: URL or local path of the git repository.
            clone_dir: Directory to clone into. Created if None.

        Returns:
            RepoInfo object with repository details and file tree.
        """
        if clone_dir is None:
            clone_dir = tempfile.mkdtemp(prefix="patchpilot_repo_")

        repo_path = Path(clone_dir)

        # Clone or pull if already cloned
        if (repo_path / ".git").exists():
            repo = git.Repo(repo_path)
            try:
                repo.remotes.origin.pull()
            except Exception:
                pass
        elif os.path.exists(repo_url) and os.path.isdir(repo_url):
            # Local directory passed instead of URL
            repo_path = Path(repo_url)
            try:
                repo = git.Repo(repo_path)
            except Exception:
                repo = None
        else:
            repo = git.Repo.clone_from(repo_url, repo_path)

        try:
            default_branch = str(repo.active_branch) if repo else "main"
        except Exception:
            default_branch = "main"

        binary_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".woff",
            ".ttf",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".exe",
            ".dll",
            ".so",
            ".pyc",
        }

        file_tree: list[str] = []
        language_stats: dict[str, int] = {}

        ext_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".html": "HTML",
            ".css": "CSS",
            ".md": "Markdown",
            ".json": "JSON",
            ".yml": "YAML",
            ".yaml": "YAML",
            ".sh": "Shell",
            ".c": "C",
            ".cpp": "C++",
            ".go": "Go",
            ".rs": "Rust",
            ".java": "Java",
        }

        for root, dirs, filenames in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            if ".venv" in dirs:
                dirs.remove(".venv")

            for filename in filenames:
                file_path = Path(root) / filename
                ext = file_path.suffix.lower()

                if ext in binary_extensions:
                    continue

                try:
                    rel_path = file_path.relative_to(repo_path).as_posix()
                    file_tree.append(rel_path)

                    lang = ext_map.get(ext, ext.lstrip(".").upper() or "Other")
                    language_stats[lang] = language_stats.get(lang, 0) + 1
                except ValueError:
                    continue

        return RepoInfo(
            clone_path=str(repo_path),
            repo_url=repo_url,
            default_branch=default_branch,
            file_tree=file_tree,
            language_stats=language_stats,
        )
