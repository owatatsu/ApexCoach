from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from apexcoach.models import Action, GameState
from apexcoach.providers.types import LLMAdviceResult, LLMProviderConfig


class BaseLLMProvider(ABC):
    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def name(self) -> str:
        """Provider display name."""

    @abstractmethod
    def healthcheck(self) -> bool:
        """Return True when provider endpoint appears healthy."""

    @abstractmethod
    def generate_advice(
        self,
        state: GameState,
        candidate_actions: Sequence[Action],
        context: dict[str, Any] | None = None,
    ) -> LLMAdviceResult | None:
        """
        Return advice result or None when provider fails/invalid.
        Implementations must never raise errors to caller.
        """
