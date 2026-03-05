from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any

from apexcoach.models import Action, GameState
from apexcoach.providers.base import BaseLLMProvider
from apexcoach.providers.types import LLMAdviceResult, LLMProviderConfig


class MockProvider(BaseLLMProvider):
    def __init__(
        self,
        config: LLMProviderConfig,
        return_none: bool = False,
        delay_ms: int = 0,
        fixed_action: Action | None = None,
    ) -> None:
        super().__init__(config)
        self.return_none = return_none
        self.delay_ms = max(0, int(delay_ms))
        self.fixed_action = fixed_action

    def name(self) -> str:
        return "mock"

    def healthcheck(self) -> bool:
        return True

    def generate_advice(
        self,
        state: GameState,
        candidate_actions: Sequence[Action],
        context: dict[str, Any] | None = None,
    ) -> LLMAdviceResult | None:
        if self.return_none:
            return None
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        candidate_list = list(candidate_actions)
        if not candidate_list:
            return None

        chosen = self.fixed_action
        if chosen is None or chosen not in candidate_list:
            chosen = candidate_list[0]

        reason = (
            f"mock advice for {chosen.value} "
            f"(hp={state.hp_pct:.2f}, shield={state.shield_pct:.2f})"
        )
        return LLMAdviceResult(
            action=chosen,
            reason=reason,
            request_id=str(uuid.uuid4()),
            provider_name=self.name(),
            model_name=self.config.model_name,
            latency_ms=float(self.delay_ms),
            raw_response=None,
        )
