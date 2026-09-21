from __future__ import annotations

from pathlib import Path

import pytest

from patchpilot.tools.dependency_analyzer import DependencyAnalyzer


@pytest.mark.asyncio
async def test_dependency_analyzer_python(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text("import os\nimport sys\nfrom pathlib import Path\n", encoding="utf-8")

    analyzer = DependencyAnalyzer()
    deps = await analyzer.analyze(str(f))
    assert "os" in deps
    assert "sys" in deps
    assert "pathlib" in deps


@pytest.mark.asyncio
async def test_dependency_graph_building(tmp_path: Path) -> None:
    f1 = tmp_path / "a.py"
    f1.write_text("import b\n", encoding="utf-8")
    f2 = tmp_path / "b.py"
    f2.write_text("import os\n", encoding="utf-8")

    analyzer = DependencyAnalyzer()
    graph = await analyzer.build_dependency_graph(str(tmp_path))
    assert "a.py" in graph or any("a.py" in k for k in graph)
