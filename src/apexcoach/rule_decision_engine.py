from __future__ import annotations

from apexcoach.config import ThresholdConfig
from apexcoach.models import Action, Decision, GameState


class RuleDecisionEngine:
    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.t = thresholds

    def decide(self, state: GameState) -> Decision:
        candidates = self.decide_candidates(state)
        if candidates:
            return candidates[0]
        return Decision(action=Action.NONE, reason="No strong signal.", confidence=0.5)

    def decide_candidates(self, state: GameState) -> list[Decision]:
        total = state.hp_pct + state.shield_pct
        vitals_reliable = state.vitals_confidence >= self.t.vitals_confidence_min
        streak_ready_for_retreat = (
            state.retreat_low_hp_streak >= self.t.low_hp_consecutive_frames
        )
        streak_ready_for_heal = (
            state.heal_low_hp_streak >= self.t.low_hp_consecutive_frames
        )
        stationary_for_heal = state.stationary_frames >= self.t.heal_stationary_frames
        stationary_for_lowhp_retreat = (
            state.stationary_frames >= self.t.retreat_lowhp_stationary_frames
        )
        safe_to_heal = (
            not state.under_fire and state.recent_damage_1s <= self.t.quiet_damage_1s
        )
        critical_heal_window = (
            safe_to_heal
            and state.stationary_frames >= max(1, self.t.heal_stationary_frames // 2)
            and (
                total <= self.t.critical_heal_total_hp_shield
                or (
                    state.shield_pct <= 0.05
                    and state.hp_pct <= self.t.broken_shield_heal_hp_pct
                )
            )
        )

        decisions: list[Decision] = []

        # Priority 1: RETREAT
        if state.ally_knock_recent and state.under_fire:
            decisions.append(
                Decision(
                action=Action.RETREAT,
                reason="Ally down while taking fire.",
                confidence=0.95,
                )
            )
        if state.allies_alive <= self.t.ally_isolated_alive_count:
            decisions.append(
                Decision(
                action=Action.RETREAT,
                reason="Isolated or almost isolated.",
                confidence=0.9,
                )
            )
        if vitals_reliable and state.recent_damage_1s >= self.t.high_damage_1s:
            decisions.append(
                Decision(
                action=Action.RETREAT,
                reason="High incoming damage in last 1s.",
                confidence=0.9,
                )
            )
        if (
            vitals_reliable
            and streak_ready_for_retreat
            and stationary_for_lowhp_retreat
            and total <= self.t.low_total_hp_shield
            and state.under_fire
        ):
            decisions.append(
                Decision(
                action=Action.RETREAT,
                reason="Low HP+Shield while under fire.",
                confidence=0.9,
                )
            )

        # Priority 2: TAKE_COVER
        if (
            state.under_fire
            and state.exposed_no_cover
            and state.exposed_confidence >= self.t.exposed_confidence_min
        ):
            decisions.append(
                Decision(
                action=Action.TAKE_COVER,
                reason="Under fire in exposed area. Move behind cover.",
                confidence=min(0.95, 0.6 + state.exposed_confidence * 0.35),
                )
            )

        # Priority 3: TAKE_HIGH_GROUND
        if (
            state.under_fire
            and state.low_ground_disadvantage
            and state.low_ground_confidence >= self.t.low_ground_confidence_min
        ):
            decisions.append(
                Decision(
                action=Action.TAKE_HIGH_GROUND,
                reason="Under fire from lower position. Take higher ground.",
                confidence=min(0.9, 0.55 + state.low_ground_confidence * 0.35),
                )
            )

        # Priority 4: HEAL
        if (
            vitals_reliable
            and streak_ready_for_heal
            and stationary_for_heal
            and safe_to_heal
            and total <= self.t.heal_total_hp_shield
        ):
            decisions.append(
                Decision(
                action=Action.HEAL,
                reason="Safe window and low HP+Shield.",
                confidence=0.8,
                )
            )
        elif critical_heal_window:
            decisions.append(
                Decision(
                    action=Action.HEAL,
                    reason="Critical low resources in a safe window.",
                    confidence=0.74,
                )
            )

        # Priority 5: PUSH
        if (
            state.enemy_knock_recent
            and state.allies_alive >= 2
            and total >= self.t.push_min_total_hp_shield
            and not state.ally_knock_recent
        ):
            decisions.append(
                Decision(
                action=Action.PUSH,
                reason="Enemy knock advantage with healthy team.",
                confidence=0.8,
                )
            )

        return _dedupe_by_action_with_priority(decisions)


def _dedupe_by_action_with_priority(candidates: list[Decision]) -> list[Decision]:
    by_action: dict[Action, Decision] = {}
    for d in candidates:
        prev = by_action.get(d.action)
        if prev is None or d.confidence > prev.confidence:
            by_action[d.action] = d

    order = [
        Action.RETREAT,
        Action.TAKE_COVER,
        Action.TAKE_HIGH_GROUND,
        Action.HEAL,
        Action.PUSH,
    ]
    return [by_action[a] for a in order if a in by_action]
