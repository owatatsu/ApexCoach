from __future__ import annotations

from apexcoach.providers.base import BaseLLMProvider
from apexcoach.providers.lmstudio_provider import LMStudioProvider
from apexcoach.providers.mock_provider import MockProvider
from apexcoach.providers.types import LLMProviderConfig


def build_provider(provider_name: str, config: LLMProviderConfig) -> BaseLLMProvider:
    name = (provider_name or "").strip().lower()
    if name == "lmstudio":
        return LMStudioProvider(config)
    if name == "mock":
        return MockProvider(config)
    raise ValueError(f"Unsupported llm provider: {provider_name!r}")
