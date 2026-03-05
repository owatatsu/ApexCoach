from __future__ import annotations

import json

from apexcoach.models import Action, GameState


def build_system_prompt(max_reason_chars: int, language: str = "ja") -> str:
    return (
        "You are an Apex Legends tactical coach.\n"
        f"Reply in {language}.\n"
        "Output JSON only.\n"
        'Schema: {"action":"<ONE_OF_CANDIDATES>","reason":"<short reason>"}\n'
        "Do not output actions outside candidate_actions.\n"
        "If uncertain, choose NONE.\n"
        f"reason must be <= {max_reason_chars} chars."
    )


def build_user_payload(
    state: GameState,
    candidate_actions: list[Action],
) -> str:
    payload = {
        "state": {
            "hp_pct": round(state.hp_pct, 3),
            "shield_pct": round(state.shield_pct, 3),
            "allies_alive": state.allies_alive,
            "allies_down": state.allies_down,
            "recent_damage_1s": round(state.recent_damage_1s, 3),
            "recent_damage_3s": round(state.recent_damage_3s, 3),
            "under_fire": state.under_fire,
            "enemy_knock_recent": state.enemy_knock_recent,
            "ally_knock_recent": state.ally_knock_recent,
            "low_ground_disadvantage": state.low_ground_disadvantage,
            "exposed_no_cover": state.exposed_no_cover,
            "is_moving": state.is_moving,
        },
        "candidate_actions": [a.value for a in candidate_actions],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_messages(
    state: GameState,
    candidate_actions: list[Action],
    max_reason_chars: int,
    language: str = "ja",
) -> list[dict[str, str]]:
    system = build_system_prompt(max_reason_chars=max_reason_chars, language=language)
    user = build_user_payload(state=state, candidate_actions=candidate_actions)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
