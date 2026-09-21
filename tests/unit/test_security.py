from __future__ import annotations

import pytest

from patchpilot.models import CodeChange
from patchpilot.tools.security import SecurityChecker


@pytest.fixture
def checker() -> SecurityChecker:
    return SecurityChecker()


@pytest.mark.asyncio
async def test_detects_hardcoded_api_key(checker: SecurityChecker) -> None:
    change = CodeChange(
        file_path="src/config.py",
        original_content="",
        modified_content='api_key = "sk-ant-api03-abcdef1234567890abcdef1234567890"',
        change_description="added key",
    )
    report = await checker.check([change])
    assert not report.passed
    assert len(report.findings) > 0


@pytest.mark.asyncio
async def test_detects_aws_key(checker: SecurityChecker) -> None:
    change = CodeChange(
        file_path="src/aws.py",
        original_content="",
        modified_content='AWS_KEY = "AKIA1234567890ABCDEF"',
        change_description="added aws key",
    )
    report = await checker.check([change])
    assert len(report.findings) > 0


@pytest.mark.asyncio
async def test_detects_private_key(checker: SecurityChecker) -> None:
    change = CodeChange(
        file_path="src/keys.py",
        original_content="",
        modified_content='key = "-----BEGIN PRIVATE KEY-----"',
        change_description="added key",
    )
    report = await checker.check([change])
    assert len(report.findings) > 0


@pytest.mark.asyncio
async def test_detects_eval(checker: SecurityChecker) -> None:
    change = CodeChange(
        file_path="src/calc.py",
        original_content="",
        modified_content='result = eval("1+1")',
        change_description="used eval",
    )
    report = await checker.check([change])
    assert len(report.findings) > 0


@pytest.mark.asyncio
async def test_detects_subprocess_shell_true(checker: SecurityChecker) -> None:
    change = CodeChange(
        file_path="src/utils.py",
        original_content="",
        modified_content='import subprocess\nsubprocess.call("ls", shell=True)',
        change_description="used shell=True",
    )
    report = await checker.check([change])
    assert len(report.findings) > 0


@pytest.mark.asyncio
async def test_clean_code_passes(checker: SecurityChecker) -> None:
    change = CodeChange(
        file_path="src/clean.py",
        original_content="",
        modified_content="x = 1\ny = 2\n",
        change_description="clean code",
    )
    report = await checker.check([change])
    assert report.passed
    assert len(report.findings) == 0


@pytest.mark.asyncio
async def test_ignores_test_files(checker: SecurityChecker) -> None:
    change = CodeChange(
        file_path="tests/test_config.py",
        original_content="",
        modified_content='api_key = "AKIA1234567890ABCDEF"',
        change_description="test fixture key",
    )
    report = await checker.check([change])
    assert report.passed


@pytest.mark.asyncio
async def test_ignores_comments(checker: SecurityChecker) -> None:
    change = CodeChange(
        file_path="src/config.py",
        original_content="",
        modified_content='# api_key = "AKIA1234567890ABCDEF"',
        change_description="commented key",
    )
    report = await checker.check([change])
    assert report.passed


@pytest.mark.asyncio
async def test_severity_determines_pass_fail(checker: SecurityChecker) -> None:
    change_clean = CodeChange(
        file_path="src/clean.py",
        original_content="",
        modified_content="def hello(): return 'world'",
        change_description="clean",
    )
    report = await checker.check([change_clean])
    assert report.passed
