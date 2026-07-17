from collections import Counter
from datetime import datetime
import time

from apexcoach.config import ApexCoachConfig
from apexcoach.display_text import format_instruction_line
from apexcoach.action_arbiter import ActionArbiter
from apexcoach.llm_advisor import AsyncAdviceResult
from apexcoach.models import Action, Decision, GameState
from apexcoach.pipeline import (
    _PipelineRuntime,
    _PipelineSession,
    RateGate,
    _build_summary,
    _is_emergency_rule_decision,
    _is_current_async_advice,
    _resolve_run_artifact_paths,
    _to_overlay_llm_message,
)


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


def test_current_async_none_advice_is_valid_for_current_rule_action() -> None:
    result = AsyncAdviceResult(
        request_id=1,
        generation=3,
        requested_timestamp=10.0,
        requested_rule_action=Action.PUSH,
        decision=Decision(Action.NONE, "hold position"),
        reason="LLM veto",
    )

    assert _is_current_async_advice(
        result,
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=10.1,
        max_age_seconds=2.0,
    ) is True


def test_to_overlay_llm_message_filters_internal_notes() -> None:
    assert _to_overlay_llm_message("llm_skip:rate_limited") is None
    assert _to_overlay_llm_message(" Use cover before healing ") == "Use cover before healing"


def test_resolve_run_artifact_paths_expands_timestamp_placeholder() -> None:
    cfg = ApexCoachConfig()

    _resolve_run_artifact_paths(cfg, now=datetime(2026, 3, 13, 12, 34, 56))

    assert cfg.logging.path == "logs/session_20260313_123456.jsonl"
    assert cfg.detection_debug.output_dir == "logs/detection_debug_20260313_123456"
    assert cfg.llm.offline_review_output == "logs/coach_review_20260313_123456.md"


def test_resolve_run_artifact_paths_keeps_explicit_paths_without_placeholder() -> None:
    cfg = ApexCoachConfig()
    cfg.logging.path = "logs/session.jsonl"
    cfg.detection_debug.output_dir = "logs/detection_debug"
    cfg.llm.offline_review_output = "logs/coach_review.md"

    _resolve_run_artifact_paths(cfg, now=datetime(2026, 3, 13, 12, 34, 56))

    assert cfg.logging.path == "logs/session.jsonl"
    assert cfg.detection_debug.output_dir == "logs/detection_debug"
    assert cfg.llm.offline_review_output == "logs/coach_review.md"


def test_format_instruction_line_localizes_for_display() -> None:
    assert format_instruction_line(Action.RETREAT, "High incoming damage in last 1s.") == (
        "退避 | 直近1秒の被ダメージが大きいです。"
    )


def test_emergency_voice_is_enqueued_before_slow_llm_path() -> None:
    order: list[str] = []

    class FakeDecisionEngine:
        def decide_candidates(self, state):
            return [Decision(Action.RETREAT, "High incoming damage in last 1s.", 0.9)]

    class FakeVoice:
        def maybe_speak(self, *, decision, arbiter, timestamp, urgent=False):
            order.append("voice")
            return "退避"

    class SlowLlm:
        def invalidate_pending_advice(self):
            order.append("invalidate")

        def poll_advice(self):
            order.append("poll")
            return None

        def maybe_advise_decision(self, **kwargs):
            order.append("llm")
            time.sleep(0.2)
            raise AssertionError("emergency rule should not wait for realtime LLM")

    class FakeAggregator:
        def record_action(self, action, timestamp):
            return None

    config = ApexCoachConfig()
    session = object.__new__(_PipelineSession)
    session.config = config
    session.decision_engine = FakeDecisionEngine()
    session.arbiter = ActionArbiter(config.arbiter)
    session.voice = FakeVoice()
    session.llm = SlowLlm()
    session.llm_gate = RateGate(config.frequencies.llm_fps)
    session.state_aggregator = FakeAggregator()
    session.runtime = _PipelineRuntime()

    state = GameState(timestamp=1.0, recent_damage_1s=0.4, under_fire=True)
    started = time.perf_counter()
    result = session._update_decision(state, timestamp=1.0)
    elapsed = time.perf_counter() - started

    assert result == "llm_skip:urgent_rule"
    assert order == ["voice", "invalidate", "poll"]
    assert elapsed < 0.15
    assert _is_emergency_rule_decision(state, Decision(Action.RETREAT, "danger"), config)


def test_pipeline_marks_only_critical_heal_as_urgent() -> None:
    config = ApexCoachConfig()
    safe_state = GameState(timestamp=1.0, hp_pct=0.45, shield_pct=0.2)
    critical_state = GameState(timestamp=1.0, hp_pct=0.15, shield_pct=0.1)

    assert not _is_emergency_rule_decision(
        safe_state,
        Decision(Action.HEAL, "Safe window and low HP+Shield."),
        config,
    )
    assert _is_emergency_rule_decision(
        critical_state,
        Decision(Action.HEAL, "Critical low resources in a safe window."),
        config,
    )


def test_async_llm_none_applies_when_rule_audio_was_emitted() -> None:
    voice_decisions: list[Action] = []

    class FakeDecisionEngine:
        def decide_candidates(self, state):
            return [Decision(Action.PUSH, "push", 0.8)]

    class FakeVoice:
        def maybe_speak(self, *, decision, arbiter, timestamp, urgent=False):
            voice_decisions.append(arbiter.action)
            return "敵ダウン、詰める"

    class FakeLlm:
        def poll_advice(self):
            return AsyncAdviceResult(
                request_id=1,
                generation=0,
                requested_timestamp=1.0,
                requested_rule_action=Action.PUSH,
                decision=Decision(Action.NONE, "hold position"),
                reason="NONE: hold position",
            )

        def request_advice_async(self, **kwargs):
            return False

    class FakeAggregator:
        def record_action(self, action, timestamp):
            return None

    config = ApexCoachConfig()
    session = object.__new__(_PipelineSession)
    session.config = config
    session.decision_engine = FakeDecisionEngine()
    session.arbiter = ActionArbiter(config.arbiter)
    session.voice = FakeVoice()
    session.llm = FakeLlm()
    session.llm_gate = RateGate(config.frequencies.llm_fps)
    session.state_aggregator = FakeAggregator()
    session.runtime = _PipelineRuntime()

    result = session._update_decision(GameState(timestamp=1.0), timestamp=1.0)

    assert result == "NONE: hold position"
    assert voice_decisions == [Action.PUSH]
    assert session.runtime.decision.action == Action.NONE
    assert session.runtime.arbiter_result.action == Action.NONE
