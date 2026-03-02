from __future__ import annotations

from apexcoach.models import ArbiterResult, Decision, GameState


class LlmAdvisor:
    """
    Optional low-frequency helper.
    MVP keeps this stub lightweight and state-driven.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def maybe_explain(
        self,
        state: GameState,
        decision: Decision,
        arbiter: ArbiterResult,
        run_now: bool,
    ) -> str | None:
        if not self.enabled or not run_now:
            return None
        return (
            f"{arbiter.action.value}: {decision.reason} "
            f"(hp={state.hp_pct:.2f}, shield={state.shield_pct:.2f})"
        )
