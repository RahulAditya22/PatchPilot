"""Security Checker — static analysis and secrets scanning for code changes.

Scans diffs for hardcoded secrets, API keys, tokens, passwords, and unsafe
code patterns like eval/exec/os.system with shell=True.

Model Attribution: Claude Opus (security-sensitive logic)
"""

from __future__ import annotations

import logging
import re

from patchpilot.models import (
    CodeChange,
    SecurityFinding,
    SecurityReport,
    Severity,
)

logger = logging.getLogger(__name__)

# Regex patterns for detecting secrets
SECRET_PATTERNS: list[tuple[str, str, Severity]] = [
    # API Keys
    (
        r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"][A-Za-z0-9_\-]{20,}['\"]",
        "Hardcoded API key",
        Severity.HIGH,
    ),
    (
        r"(?i)(secret[_-]?key|secretkey)\s*[=:]\s*['\"][A-Za-z0-9_\-]{20,}['\"]",
        "Hardcoded secret key",
        Severity.CRITICAL,
    ),
    # AWS
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID", Severity.CRITICAL),
    (
        r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"][A-Za-z0-9/+=]{40}['\"]",
        "AWS Secret Access Key",
        Severity.CRITICAL,
    ),
    # GitHub
    (r"gh[pousr]_[A-Za-z0-9_]{36,}", "GitHub Personal Access Token", Severity.CRITICAL),
    # Generic tokens
    (
        r"(?i)(token|bearer)\s*[=:]\s*['\"][A-Za-z0-9_\-\.]{20,}['\"]",
        "Hardcoded token",
        Severity.HIGH,
    ),
    # Passwords
    (
        r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
        "Hardcoded password",
        Severity.CRITICAL,
    ),
    # Private keys
    (r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----", "Private key", Severity.CRITICAL),
    # Connection strings
    (
        r"(?i)(mysql|postgres|mongodb|redis)://[^\s'\"]+:[^\s'\"]+@",
        "Database connection string with credentials",
        Severity.HIGH,
    ),
    # Google Cloud
    (r"AIza[0-9A-Za-z_\-]{35}", "Google API Key", Severity.HIGH),
    # Anthropic
    (r"sk-ant-[A-Za-z0-9_\-]{40,}", "Anthropic API Key", Severity.CRITICAL),
]

# Patterns for unsafe code
UNSAFE_CODE_PATTERNS: list[tuple[str, str, Severity]] = [
    (r"\beval\s*\(", "Use of eval() — potential code injection", Severity.HIGH),
    (r"\bexec\s*\(", "Use of exec() — potential code injection", Severity.HIGH),
    (r"os\.system\s*\(", "Use of os.system() — prefer subprocess", Severity.MEDIUM),
    (
        r"subprocess\.call\([^)]*shell\s*=\s*True",
        "subprocess with shell=True — command injection risk",
        Severity.HIGH,
    ),
    (
        r"subprocess\.run\([^)]*shell\s*=\s*True",
        "subprocess.run with shell=True — command injection risk",
        Severity.HIGH,
    ),
    (
        r"subprocess\.Popen\([^)]*shell\s*=\s*True",
        "subprocess.Popen with shell=True — command injection risk",
        Severity.HIGH,
    ),
    (r"__import__\s*\(", "Dynamic import with __import__() — security concern", Severity.MEDIUM),
    (
        r"pickle\.loads?\s*\(",
        "Pickle deserialization — potential arbitrary code execution",
        Severity.HIGH,
    ),
    (
        r"yaml\.load\s*\([^)]*\)",
        "yaml.load without SafeLoader — potential code execution",
        Severity.MEDIUM,
    ),
    (r"marshal\.loads?\s*\(", "Marshal deserialization — potential code execution", Severity.HIGH),
]


class SecurityChecker:
    """Checks code changes for security issues.

    Performs two categories of checks:
    1. Secrets scanning: Detects hardcoded credentials, API keys, tokens
    2. Unsafe code detection: Flags dangerous function calls

    All checks are purely static (regex-based) and do not execute code.
    """

    def __init__(self, extra_patterns: list[tuple[str, str, Severity]] | None = None) -> None:
        """Initialize the security checker.

        Args:
            extra_patterns: Optional additional regex patterns to check.
                Each tuple is (pattern, description, severity).
        """
        self._secret_patterns = list(SECRET_PATTERNS)
        self._unsafe_patterns = list(UNSAFE_CODE_PATTERNS)
        if extra_patterns:
            self._secret_patterns.extend(extra_patterns)

    async def check(self, changes: list[CodeChange]) -> SecurityReport:
        """Run all security checks on the given code changes.

        Args:
            changes: List of code changes to scan.

        Returns:
            SecurityReport with pass/fail status and all findings.
        """
        findings: list[SecurityFinding] = []

        for change in changes:
            # Only scan the new/modified content
            content = change.modified_content
            if not content:
                continue

            # Secrets scanning
            secret_findings = self._scan_for_secrets(change.file_path, content)
            findings.extend(secret_findings)

            # Unsafe code scanning (only for code files)
            if self._is_code_file(change.file_path):
                unsafe_findings = self._scan_for_unsafe_code(change.file_path, content)
                findings.extend(unsafe_findings)

        # Determine pass/fail: fail on any HIGH or CRITICAL
        has_blocking = any(f.severity in (Severity.HIGH, Severity.CRITICAL) for f in findings)

        passed = not has_blocking

        if findings:
            logger.warning(
                "Security check: %d findings (%s)",
                len(findings),
                "BLOCKING" if has_blocking else "advisory",
            )
        else:
            logger.info("Security check: no issues found")

        return SecurityReport(passed=passed, findings=findings)

    def _scan_for_secrets(self, file_path: str, content: str) -> list[SecurityFinding]:
        """Scan content for hardcoded secrets.

        Args:
            file_path: Path of the file being scanned.
            content: File content to scan.

        Returns:
            List of security findings for secrets.
        """
        findings: list[SecurityFinding] = []
        lines = content.splitlines()

        for pattern_str, description, severity in self._secret_patterns:
            try:
                pattern = re.compile(pattern_str)
            except re.error:
                logger.warning("Invalid regex pattern: %s", pattern_str)
                continue

            for line_num, line in enumerate(lines, 1):
                # Skip comments and test files
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue

                if pattern.search(line):
                    # Check if it's in a test/example/fixture context
                    if self._is_test_or_example(file_path, line):
                        continue

                    findings.append(
                        SecurityFinding(
                            severity=severity,
                            finding_type="secret",
                            file_path=file_path,
                            line_number=line_num,
                            description=f"{description}: detected on line {line_num}",
                            recommendation=(
                                "Move this value to an environment variable or "
                                "secrets manager. Never commit credentials to source control."
                            ),
                        )
                    )

        return findings

    def _scan_for_unsafe_code(self, file_path: str, content: str) -> list[SecurityFinding]:
        """Scan content for unsafe code patterns.

        Args:
            file_path: Path of the file being scanned.
            content: File content to scan.

        Returns:
            List of security findings for unsafe code.
        """
        findings: list[SecurityFinding] = []
        lines = content.splitlines()

        for pattern_str, description, severity in self._unsafe_patterns:
            try:
                pattern = re.compile(pattern_str)
            except re.error:
                continue

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                if pattern.search(line):
                    findings.append(
                        SecurityFinding(
                            severity=severity,
                            finding_type="unsafe_code",
                            file_path=file_path,
                            line_number=line_num,
                            description=f"{description} (line {line_num})",
                            recommendation=(
                                "Review this code for security implications. "
                                "Consider using a safer alternative."
                            ),
                        )
                    )

        return findings

    @staticmethod
    def _is_code_file(file_path: str) -> bool:
        """Check if a file is a source code file.

        Args:
            file_path: File path to check.

        Returns:
            True if the file is a code file.
        """
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".rb",
            ".rs",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".php",
            ".sh",
            ".bash",
            ".zsh",
        }
        for ext in code_extensions:
            if file_path.endswith(ext):
                return True
        return False

    @staticmethod
    def _is_test_or_example(file_path: str, line: str) -> bool:
        """Check if a finding is likely in a test or example context.

        Args:
            file_path: File path.
            line: The matching line.

        Returns:
            True if likely a test/example (reduce false positives).
        """
        test_indicators = [
            "test_",
            "_test.",
            "tests/",
            "test/",
            "example",
            "fixture",
            "mock",
            "fake",
            "dummy",
            "sample",
        ]
        file_lower = file_path.lower()
        for indicator in test_indicators:
            if indicator in file_lower:
                return True

        # Check for obvious test/example values in the line
        line_lower = line.lower()
        fake_values = [
            "test-key",
            "fake-key",
            "dummy",
            "example",
            "xxx",
            "your-",
            "replace-",
            "placeholder",
            "sk-test-",
            "pk-test-",
        ]
        for fake in fake_values:
            if fake in line_lower:
                return True

        return False
