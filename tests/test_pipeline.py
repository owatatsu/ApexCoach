from collections import Counter

from apexcoach.display_text import format_instruction_line
from apexcoach.models import Action
from apexcoach.pipeline import RateGate, _build_summary, _to_overlay_llm_message


def test_rate_gate_respects_interval() -> None:
    gate = RateGate(fps=2)

    assert gate.ready(0.0) is True
    assert gate.ready(0.1) is False
    assert gate.ready(0.5) is True


def test_build_summary_includes_all_actions() -> None:
    summary = _build_summary(
        frames=7,
        action_counts=Counter({"HEAL": 2, "PUSH": 1}),
    )

    assert summary == {
        "frames": 7,
        "NONE": 0,
        "HEAL": 2,
        "RETREAT": 0,
        "TAKE_COVER": 0,
        "TAKE_HIGH_GROUND": 0,
        "PUSH": 1,
    }


def test_to_overlay_llm_message_filters_internal_notes() -> None:
    assert _to_overlay_llm_message("llm_skip:rate_limited") is None
    assert _to_overlay_llm_message(" Use cover before healing ") == "Use cover before healing"


def test_format_instruction_line_localizes_for_display() -> None:
    assert format_instruction_line(Action.RETREAT, "High incoming damage in last 1s.") == (
        "退避 | 直近1秒の被ダメージが大きいです。"
    )
