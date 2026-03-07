from __future__ import annotations

import json
from typing import Any

from apexcoach.models import Action, GameState


def build_system_prompt(max_reason_chars: int, language: str = "ja") -> str:
    return (
        "You are an Apex Legends tactical coach.\n"
        f"Reply in {language}.\n"
        "Output JSON only.\n"
        'Schema: {"action":"<ONE_OF_CANDIDATES>","reason":"<short reason>"}\n'
        "Do not output actions outside candidate_actions.\n"
        "Use state, candidate_details, and rule_decision as your main evidence.\n"
        "If uncertain, choose NONE.\n"
        f"reason must be <= {max_reason_chars} chars."
    )


def build_user_payload(
    state: GameState,
    candidate_actions: list[Action],
    context: dict[str, Any] | None = None,
) -> str:
    context = context or {}
    payload = {
        "state": {
            "timestamp": round(state.timestamp, 3),
            "hp_pct": round(state.hp_pct, 3),
            "shield_pct": round(state.shield_pct, 3),
            "total_hp_shield": round(state.hp_pct + state.shield_pct, 3),
            "vitals_confidence": round(state.vitals_confidence, 3),
            "allies_alive": state.allies_alive,
            "allies_down": state.allies_down,
            "recent_damage_1s": round(state.recent_damage_1s, 3),
            "recent_damage_3s": round(state.recent_damage_3s, 3),
            "under_fire": state.under_fire,
            "enemy_knock_recent": state.enemy_knock_recent,
            "ally_knock_recent": state.ally_knock_recent,
            "low_ground_disadvantage": state.low_ground_disadvantage,
            "low_ground_confidence": round(state.low_ground_confidence, 3),
            "exposed_no_cover": state.exposed_no_cover,
            "exposed_confidence": round(state.exposed_confidence, 3),
            "is_moving": state.is_moving,
            "movement_score": round(state.movement_score, 3),
            "moving_recent_frames": state.moving_recent_frames,
            "stationary_frames": state.stationary_frames,
            "last_action": state.last_action.value,
            "last_action_time": (
                round(state.last_action_time, 3)
                if state.last_action_time is not None
                else None
            ),
        },
        "candidate_actions": [a.value for a in candidate_actions],
        "candidate_details": context.get("candidate_details", []),
        "rule_decision": context.get("rule_decision"),
        "request_context": {
            "request_timestamp": context.get("timestamp"),
            "decision_trigger": context.get("decision_trigger"),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_messages(
    state: GameState,
    candidate_actions: list[Action],
    max_reason_chars: int,
    language: str = "ja",
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    system = build_system_prompt(max_reason_chars=max_reason_chars, language=language)
    user = build_user_payload(
        state=state,
        candidate_actions=candidate_actions,
        context=context,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
