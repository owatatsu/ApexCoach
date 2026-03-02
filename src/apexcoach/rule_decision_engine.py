from __future__ import annotations

from apexcoach.config import ThresholdConfig
from apexcoach.models import Action, Decision, GameState


class RuleDecisionEngine:
    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.t = thresholds

    def decide(self, state: GameState) -> Decision:
        total = state.hp_pct + state.shield_pct

        # Priority 1: RETREAT
        if state.ally_knock_recent and state.under_fire:
            return Decision(
                action=Action.RETREAT,
                reason="Ally down while taking fire.",
                confidence=0.95,
            )
        if state.allies_alive <= self.t.ally_isolated_alive_count:
            return Decision(
                action=Action.RETREAT,
                reason="Isolated or almost isolated.",
                confidence=0.9,
            )
        if state.recent_damage_1s >= self.t.high_damage_1s:
            return Decision(
                action=Action.RETREAT,
                reason="High incoming damage in last 1s.",
                confidence=0.9,
            )
        if total <= self.t.low_total_hp_shield and state.under_fire:
            return Decision(
                action=Action.RETREAT,
                reason="Low HP+Shield while under fire.",
                confidence=0.9,
            )

        # Priority 2: HEAL
        if (
            not state.under_fire
            and total <= self.t.heal_total_hp_shield
            and state.recent_damage_1s <= self.t.quiet_damage_1s
        ):
            return Decision(
                action=Action.HEAL,
                reason="Safe window and low HP+Shield.",
                confidence=0.8,
            )

        # Priority 3: PUSH
        if (
            state.enemy_knock_recent
            and state.allies_alive >= 2
            and total >= self.t.push_min_total_hp_shield
            and not state.ally_knock_recent
        ):
            return Decision(
                action=Action.PUSH,
                reason="Enemy knock advantage with healthy team.",
                confidence=0.8,
            )

        return Decision(action=Action.NONE, reason="No strong signal.", confidence=0.5)
