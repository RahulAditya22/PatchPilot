"""Unified async LLM client supporting Anthropic Claude and Google Gemini.

Provides a consistent interface for text generation, structured output via
tool-calling, and token tracking across multiple LLM providers.

Model Attribution: Claude Opus (prompt engineering and interface design)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from patchpilot.config import get_config
from patchpilot.models import TokenUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "ANTHROPIC"
    GOOGLE = "GOOGLE"


class LLMResponse(BaseModel):
    """Unified response from any LLM provider."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    token_usage: TokenUsage
    raw_response: Any = None


# Cost per 1M tokens (input, output) in USD
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate API call cost in USD.

    Args:
        model: Model name.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    # Find matching cost entry
    for model_key, (input_cost, output_cost) in MODEL_COSTS.items():
        if model_key in model:
            return (input_tokens / 1_000_000) * input_cost + (
                output_tokens / 1_000_000
            ) * output_cost

    # Default fallback
    return (input_tokens / 1_000_000) * 1.0 + (output_tokens / 1_000_000) * 5.0


class LLMClient:
    """Async LLM client supporting Anthropic and Google Gemini.

    Provides unified generate() and generate_structured() methods with
    automatic retry, token tracking, and cost estimation.
    """

    def __init__(self, provider: LLMProvider, model: str, api_key: str) -> None:
        """Initialize the LLM client.

        Args:
            provider: The LLM provider to use.
            model: Model identifier string.
            api_key: API key for authentication.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._anthropic_client: Any = None
        self._google_client: Any = None

        if self.provider == LLMProvider.ANTHROPIC:
            from anthropic import AsyncAnthropic

            self._anthropic_client = AsyncAnthropic(api_key=api_key)
        elif self.provider == LLMProvider.GOOGLE:
            from google import genai

            self._google_client = genai.Client(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation messages with 'role' and 'content' keys.
            system_prompt: System instruction for the model.
            tools: Optional tool schemas for function calling.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum output tokens.

        Returns:
            Unified LLMResponse with content, tool calls, and usage.
        """
        if self.provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic(
                messages, system_prompt, tools, temperature, max_tokens
            )
        return await self._generate_google(messages, system_prompt, tools, temperature, max_tokens)

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        response_schema: type[T],
        temperature: float = 0.0,
    ) -> T:
        """Generate structured output conforming to a Pydantic model.

        For Anthropic: uses tool_use with a schema matching the model.
        For Google: uses response_mime_type='application/json' with response_schema.

        Args:
            messages: Conversation messages.
            system_prompt: System instruction.
            response_schema: Pydantic model class to validate against.
            temperature: Sampling temperature.

        Returns:
            Instance of the response_schema Pydantic model.

        Raises:
            ValueError: If the model fails to return valid structured data.
        """
        if self.provider == LLMProvider.ANTHROPIC:
            return await self._structured_anthropic(
                messages, system_prompt, response_schema, temperature
            )
        return await self._structured_google(messages, system_prompt, response_schema, temperature)

    async def _generate_anthropic(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate using Anthropic Claude API.

        Args:
            messages: Conversation messages.
            system_prompt: System prompt.
            tools: Tool schemas.
            temperature: Temperature.
            max_tokens: Max output tokens.

        Returns:
            LLMResponse.
        """
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        response = await self._anthropic_client.messages.create(**kwargs)

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if isinstance(block, anthropic.types.TextBlock):
                content += block.text
            elif isinstance(block, anthropic.types.ToolUseBlock):
                tool_calls.append(
                    {
                        "name": block.name,
                        "arguments": block.input,
                        "id": block.id,
                    }
                )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        usage = TokenUsage(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            timestamp=datetime.now(UTC),
        )

        return LLMResponse(
            content=content.strip(),
            tool_calls=tool_calls if tool_calls else None,
            token_usage=usage,
            raw_response=response,
        )

    async def _generate_google(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate using Google Gemini API.

        Args:
            messages: Conversation messages.
            system_prompt: System prompt.
            tools: Tool schemas (logged but not mapped for general generate).
            temperature: Temperature.
            max_tokens: Max output tokens.

        Returns:
            LLMResponse.
        """
        from google.genai import types as genai_types

        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=msg["content"])],
                )
            )

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        if tools:
            logger.debug(
                "Tool calling for Google Gemini in general generate is simplified; "
                "use generate_structured for schema-constrained output."
            )

        config = genai_types.GenerateContentConfig(**config_kwargs)

        response = await self._google_client.aio.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )

        content = response.text or ""
        input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        output_tokens = (
            response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        )

        usage = TokenUsage(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            timestamp=datetime.now(UTC),
        )

        return LLMResponse(
            content=content.strip(),
            tool_calls=None,
            token_usage=usage,
            raw_response=response,
        )

    async def _structured_anthropic(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        response_schema: type[T],
        temperature: float,
    ) -> T:
        """Get structured output from Anthropic via tool_use.

        Args:
            messages: Messages.
            system_prompt: System prompt.
            response_schema: Target Pydantic model.
            temperature: Temperature.

        Returns:
            Validated instance of response_schema.
        """
        json_schema = response_schema.model_json_schema()
        tool = {
            "name": "extract_structured_data",
            "description": "Extract structured data based on the provided schema.",
            "input_schema": {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
            },
        }

        augmented_prompt = (
            system_prompt
            + "\n\nYou must use the `extract_structured_data` tool to return your response."
        )

        response = await self._generate_anthropic(
            messages=messages,
            system_prompt=augmented_prompt,
            tools=[tool],
            temperature=temperature,
            max_tokens=4096,
        )

        if response.tool_calls and len(response.tool_calls) > 0:
            arguments = response.tool_calls[0]["arguments"]
            return response_schema.model_validate(arguments)

        raise ValueError("Anthropic model failed to return structured data via tool call.")

    async def _structured_google(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        response_schema: type[T],
        temperature: float,
    ) -> T:
        """Get structured output from Google Gemini via JSON mode.

        Args:
            messages: Messages.
            system_prompt: System prompt.
            response_schema: Target Pydantic model.
            temperature: Temperature.

        Returns:
            Validated instance of response_schema.
        """
        from google.genai import types as genai_types

        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=msg["content"])],
                )
            )

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema.model_json_schema(),
        )

        response = await self._google_client.aio.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )

        data = json.loads(response.text or "{}")
        return response_schema.model_validate(data)


def create_client(provider: str, model: str | None = None) -> LLMClient:
    """Factory function to create an LLMClient from configuration.

    Args:
        provider: Provider string ('ANTHROPIC' or 'GOOGLE').
        model: Optional model name override.

    Returns:
        Configured LLMClient instance.

    Raises:
        ValueError: If the provider is unknown or API key is missing.
    """
    config = get_config()
    provider_enum = LLMProvider(provider.upper())

    if provider_enum == LLMProvider.ANTHROPIC:
        api_key = config.ANTHROPIC_API_KEY
        selected_model = model or config.DEFAULT_OPUS_MODEL
    elif provider_enum == LLMProvider.GOOGLE:
        api_key = config.GOOGLE_API_KEY
        selected_model = model or config.DEFAULT_GEMINI_MODEL
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if not api_key:
        raise ValueError(f"API key for {provider} is not configured. Set it in .env.")

    return LLMClient(provider=provider_enum, model=selected_model, api_key=api_key)
