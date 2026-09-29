"""Factory for creating LLM instances."""

from ..core.exceptions import LLMError
from .base import BaseLLM
from .claude import ClaudeLLM
from .claude_code import ClaudeCodeLLM
from .openai_llm import OpenAILLM


def create_llm(
    provider: str,
    api_key: str,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    cli_path: str = "claude",
) -> BaseLLM:
    """Create an LLM instance based on provider.

    Args:
        provider: LLM provider name (claude, claude_code, openai).
        api_key: API key for the provider (unused by claude_code).
        model: Optional model override.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        cli_path: Claude Code executable, used by claude_code.

    Returns:
        LLM instance.

    Raises:
        LLMError: If provider is unknown.
    """
    provider = provider.lower()

    if provider == "claude":
        return ClaudeLLM(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif provider == "claude_code":
        return ClaudeCodeLLM(model=model or "sonnet", cli_path=cli_path)
    elif provider == "openai":
        return OpenAILLM(
            api_key=api_key,
            model=model or "gpt-4.1-mini",
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise LLMError(f"Unknown LLM provider: {provider}. Supported: claude, claude_code, openai")
