"""Issue parser — parses GitHub issue body into structured requirements.

Model Attribution: Gemini (tool wiring)
"""

from __future__ import annotations

import re

from patchpilot.models import IssueRequirement


class IssueParser:
    """Parses issue text into structured requirements."""

    async def parse(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> IssueRequirement:
        """Parse issue text into structured requirements.

        Args:
            title: The issue title.
            body: The issue body text.
            labels: List of labels attached to the issue.

        Returns:
            IssueRequirement: Structured requirement object.
        """
        labels = labels or []
        requirements: list[str] = []

        # Extract code blocks
        code_block_pattern = re.compile(r"```.*?\n(.*?)\n```", re.DOTALL)

        # Remove code blocks from body for requirement parsing
        body_no_code = code_block_pattern.sub("", body)

        # Extract bullet points and numbered lists
        list_pattern = re.compile(r"^(?:-|\*|\d+\.)\s+(.+)$", re.MULTILINE)
        for match in list_pattern.finditer(body_no_code):
            req_text = match.group(1).strip()
            if req_text:
                requirements.append(req_text)

        # Detect priority
        priority = "medium"
        priority_keywords = {"critical", "urgent", "high", "low", "medium"}

        for label in labels:
            if label.lower() in priority_keywords:
                priority = label.lower()
                break

        if priority == "medium":
            lower_combined = f"{title} {body}".lower()
            for kw in ["critical", "urgent", "high", "low"]:
                if kw in lower_combined:
                    priority = kw
                    break

        return IssueRequirement(
            title=title,
            description=body,
            requirements=requirements,
            labels=labels,
            priority=priority,
        )
