from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from patchpilot.llm.client import LLMClient, LLMProvider, _estimate_cost, create_client


def test_estimate_cost() -> None:
    cost = _estimate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
    assert cost == 18.0  # $3 + $15

    cost_gemini = _estimate_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
    assert cost_gemini == 0.50  # $0.10 + $0.40


def test_create_client() -> None:
    with patch("patchpilot.llm.client.get_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            ANTHROPIC_API_KEY="test_anthropic",
            GOOGLE_API_KEY="test_google",
            DEFAULT_OPUS_MODEL="claude-sonnet-4-20250514",
            DEFAULT_GEMINI_MODEL="gemini-2.0-flash",
        )
        with patch("anthropic.AsyncAnthropic"):
            client = create_client("ANTHROPIC")
            assert client.provider == LLMProvider.ANTHROPIC

        with patch("google.genai.Client"):
            client_g = create_client("GOOGLE")
            assert client_g.provider == LLMProvider.GOOGLE


@pytest.mark.asyncio
async def test_generate_anthropic() -> None:
    with patch("anthropic.AsyncAnthropic") as mock_anth_cls:
        mock_anth = AsyncMock()
        mock_block = anthropic.types.TextBlock(text="Hello world", type="text")

        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage.input_tokens = 10
        mock_resp.usage.output_tokens = 5
        mock_anth.messages.create = AsyncMock(return_value=mock_resp)
        mock_anth_cls.return_value = mock_anth

        client = LLMClient(LLMProvider.ANTHROPIC, "claude-sonnet-4-20250514", "key")
        resp = await client.generate([{"role": "user", "content": "hi"}], system_prompt="sys")
        assert resp.content == "Hello world"
        assert resp.token_usage.input_tokens == 10
