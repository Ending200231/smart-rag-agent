"""Unified LLM interface supporting multiple providers."""

from langchain_openai import ChatOpenAI

from smart_rag.config import LLMConfig


def get_llm(config: LLMConfig | None = None) -> ChatOpenAI:
    """Create a ChatModel instance based on provider config.

    All supported providers (DeepSeek, Zhipu, Qwen) use OpenAI-compatible APIs,
    so we use ChatOpenAI with different base_url and api_key.
    """
    if config is None:
        config = LLMConfig()

    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
