import json

from apexcoach.models import GameState
from apexcoach.providers.prompt_builder import build_user_payload


def test_build_user_payload_omits_enemy_fields_when_unavailable() -> None:
    payload = json.loads(build_user_payload(GameState(timestamp=1.0), candidate_actions=[]))

    assert "enemy_summary" not in payload
    assert "enemy_count" not in payload["state"]


def test_build_user_payload_includes_enemy_summary_when_available() -> None:
    state = GameState(
        timestamp=1.0,
        enemy_available=True,
        enemy_count=2,
        enemy_left=1,
        enemy_center=1,
        enemy_right=0,
        tracked_enemy_ids=[3, 7],
        enemy_movement_trend="left_to_right",
        enemy_summary_lines=[
            "enemy_count=2",
            "enemy_left=1",
            "enemy_center=1",
            "tracked_enemy_ids=[3, 7]",
        ],
    )

    payload = json.loads(build_user_payload(state, candidate_actions=[]))

    assert payload["state"]["enemy_count"] == 2
    assert payload["state"]["tracked_enemy_ids"] == [3, 7]
    assert "enemy_count=2" in payload["enemy_summary"]


def test_build_user_payload_includes_low_ground_evidence() -> None:
    state = GameState(
        timestamp=1.0,
        low_ground_disadvantage=True,
        low_ground_confidence=0.72,
        low_ground_evidence=["horizon_pitch", "enemy_high_in_frame:0.61"],
    )

    payload = json.loads(build_user_payload(state, candidate_actions=[]))

    assert payload["state"]["low_ground_evidence"] == [
        "horizon_pitch",
        "enemy_high_in_frame:0.61",
    ]
