from __future__ import annotations

import sys
from pathlib import Path

import pytest

from patchpilot.tools.sandbox import SandboxRunner


@pytest.mark.asyncio
async def test_run_command_success(tmp_path: Path) -> None:
    """Run 'python -c "print(42)"', verify exit_code=0."""
    runner = SandboxRunner()
    exit_code, stdout, stderr = await runner.run_command(
        f'"{sys.executable}" -c "print(42)"', cwd=str(tmp_path)
    )
    assert exit_code == 0
    assert "42" in stdout


@pytest.mark.asyncio
async def test_run_command_failure(tmp_path: Path) -> None:
    """Run a failing command, verify non-zero exit code."""
    runner = SandboxRunner()
    exit_code, stdout, stderr = await runner.run_command(
        f'"{sys.executable}" -c "raise Exception()"', cwd=str(tmp_path)
    )
    assert exit_code != 0


def test_parse_pytest_output() -> None:
    """Parse typical pytest summary line."""
    runner = SandboxRunner()
    result = runner._parse_pytest_output("=== 1 failed, 2 passed in 0.12s ===", "", 1, 0.12)
    assert result.num_failed == 1
    assert result.num_passed == 2
    assert not result.passed


def test_detect_test_command(tmp_path: Path) -> None:
    """Verify detection from pyproject.toml presence."""
    (tmp_path / "pyproject.toml").touch()
    runner = SandboxRunner()
    cmd = runner._detect_test_command(str(tmp_path))
    assert "pytest" in cmd
