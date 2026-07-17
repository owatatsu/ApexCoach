import json
import time
from threading import Event
from urllib import error

from apexcoach.config import LlmConfig
from apexcoach.llm_advisor import (
    AsyncAdviceResult,
    LlmAdvisor,
    _build_advice_context,
    _build_review_payload,
)
from apexcoach.models import Action, ArbiterResult, Decision, GameState


def test_maybe_explain_disabled_by_default() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=False))
    result = advisor.maybe_explain(
        state=GameState(timestamp=1.0),
        decision=Decision(action=Action.NONE, reason="none"),
        arbiter=ArbiterResult(
            action=Action.NONE,
            emitted=False,
            reason="none",
            source_action=Action.NONE,
        ),
        run_now=True,
    )
    assert result is None


def test_build_advice_context_includes_candidates_and_trigger() -> None:
    state = GameState(
        timestamp=5.0,
        hp_pct=0.4,
        shield_pct=0.3,
        recent_damage_1s=0.15,
        under_fire=True,
    )
    context = _build_advice_context(
        state=state,
        candidates=[
            Decision(action=Action.RETREAT, reason="High incoming damage in last 1s.", confidence=0.9),
            Decision(action=Action.NONE, reason="No strong signal.", confidence=0.5),
        ],
        rule_decision=Decision(
            action=Action.RETREAT,
            reason="High incoming damage in last 1s.",
            confidence=0.9,
        ),
        timestamp=5.25,
    )

    assert context["decision_trigger"] == "damage_spike"
    assert context["timestamp"] == 5.25
    assert context["rule_decision"]["action"] == "RETREAT"
    assert context["candidate_details"][1]["action"] == "NONE"


def test_build_review_payload_extracts_timeline(tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    record = {
        "timestamp": 12.5,
        "frame_index": 10,
        "state": {
            "hp_pct": 0.5,
            "shield_pct": 0.3,
            "under_fire": True,
            "allies_alive": 2,
            "allies_down": 1,
            "recent_damage_1s": 0.2,
            "low_ground_disadvantage": True,
            "exposed_no_cover": False,
        },
        "events": {"damage_delta": 0.2},
        "decision": {"reason": "High incoming damage in last 1s."},
        "arbiter": {"action": "RETREAT", "emitted": True},
    }
    log_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = _build_review_payload(
        log_path,
        max_events=16,
        reason_max_chars=96,
        summary={"frames": 1},
    )
    assert payload is not None
    assert payload["meta"]["frames"] == 1
    assert payload["timeline"][0]["action"] == "RETREAT"
    assert payload["timeline"][0]["under_fire"] is True


def test_build_review_payload_downsamples_timeline_across_full_session(tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    rows = []
    for i in range(8):
        rows.append(
            {
                "timestamp": float(i * 10),
                "frame_index": i,
                "state": {
                    "hp_pct": 0.8,
                    "shield_pct": 0.5,
                    "under_fire": bool(i % 2),
                    "allies_alive": 3,
                    "allies_down": 0,
                },
                "events": {"damage_delta": 0.1},
                "decision": {"reason": f"reason-{i}"},
                "arbiter": {"action": "TAKE_COVER", "emitted": True},
            }
        )
    log_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = _build_review_payload(
        log_path,
        max_events=4,
        reason_max_chars=96,
        summary={"frames": 8},
    )

    assert payload is not None
    timeline = payload["timeline"]
    assert len(timeline) == 4
    assert timeline[0]["t"] == 0.0
    assert timeline[-1]["t"] == 70.0


def test_build_review_payload_excludes_voice_events_from_frame_statistics(tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    records = [
        {
            "timestamp": 0.0,
            "state": {"hp_pct": 0.9, "shield_pct": 0.8, "under_fire": True},
            "decision": {"reason": "cover"},
            "arbiter": {"action": "TAKE_COVER", "emitted": True},
        },
        {
            "timestamp": 10.0,
            "state": {"hp_pct": 0.2, "shield_pct": 0.2, "under_fire": False},
            "decision": {"reason": "heal"},
            "arbiter": {"action": "HEAL", "emitted": True},
        },
        {
            "record_type": "voice_event",
            "event": "completed",
            "action": "HEAL",
            "text": "今、回復",
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    payload = _build_review_payload(
        log_path,
        max_events=16,
        reason_max_chars=96,
        summary={"frames": 2},
    )

    assert payload is not None
    assert payload["meta"]["frames"] == 2
    assert payload["meta"]["duration_sec"] == 10.0
    assert payload["ratios"]["under_fire_ratio"] == 0.5
    assert payload["ratios"]["low_resource_ratio"] == 0.5
    assert len(payload["timeline"]) == 2


def test_async_advice_is_not_published_after_invalidation() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=True, provider="mock"))
    started = Event()
    release = Event()

    def slow_advice(**kwargs):
        started.set()
        release.wait(timeout=1.0)
        return Decision(Action.PUSH, "old push"), "old result"

    advisor.maybe_advise_decision = slow_advice
    assert advisor.request_advice_async(
        state=GameState(timestamp=1.0),
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=1.0,
        run_now=True,
    )
    assert started.wait(timeout=1.0)
    advisor.invalidate_pending_advice()
    release.set()
    time.sleep(0.05)
    assert advisor.poll_advice() is None
    advisor.close()


def test_async_advice_completed_after_close_is_not_published() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=True, provider="mock"))
    started = Event()
    release = Event()

    def slow_advice(**kwargs):
        started.set()
        release.wait(timeout=1.0)
        return Decision(Action.PUSH, "old push"), "old result"

    advisor.maybe_advise_decision = slow_advice
    assert advisor.request_advice_async(
        state=GameState(timestamp=1.0),
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=1.0,
        run_now=True,
    )
    assert started.wait(timeout=1.0)
    advisor.close()
    release.set()
    time.sleep(0.05)
    assert advisor.poll_advice() is None


def test_async_request_is_not_started_when_llm_is_disabled() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=False))

    assert not advisor.request_advice_async(
        state=GameState(timestamp=1.0),
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=1.0,
        run_now=True,
    )
    assert advisor._async_thread is None
    assert advisor._next_request_id == 0
    assert advisor._async_generation == 0


def test_async_request_is_not_started_when_advice_is_disabled() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=True, advice_enabled=False))

    assert not advisor.request_advice_async(
        state=GameState(timestamp=1.0),
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=1.0,
        run_now=True,
    )
    assert advisor._async_thread is None
    assert advisor._next_request_id == 0
    assert advisor._async_generation == 0


def test_rejected_async_request_preserves_completed_result() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=True, advice_enabled=False))
    completed = AsyncAdviceResult(
        request_id=7,
        generation=2,
        requested_timestamp=1.0,
        requested_rule_action=Action.PUSH,
        decision=Decision(Action.NONE, "hold"),
        reason="completed",
    )
    advisor._async_done = True
    advisor._async_result = completed
    advisor._next_request_id = 7
    advisor._async_generation = 2

    assert not advisor.request_advice_async(
        state=GameState(timestamp=2.0),
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=2.0,
        run_now=True,
    )
    assert advisor.poll_advice() == completed
    assert advisor._next_request_id == 7
    assert advisor._async_generation == 2


def test_async_request_is_not_started_without_advice_provider() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=True, provider="unsupported"))

    assert not advisor.request_advice_async(
        state=GameState(timestamp=1.0),
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=1.0,
        run_now=True,
    )
    assert advisor._async_thread is None
    assert advisor._next_request_id == 0
    assert advisor._async_generation == 0


def test_async_request_is_not_started_after_close() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=True, provider="mock"))
    advisor.close()

    assert not advisor.request_advice_async(
        state=GameState(timestamp=1.0),
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=1.0,
        run_now=True,
    )
    assert advisor._async_thread is None
    assert advisor._next_request_id == 0
    assert advisor._async_generation == 1


def test_async_request_starts_with_mock_provider() -> None:
    advisor = LlmAdvisor(LlmConfig(enabled=True, provider="mock"))

    assert advisor.request_advice_async(
        state=GameState(timestamp=1.0, recent_damage_1s=0.2),
        candidates=[Decision(Action.PUSH, "push")],
        rule_decision=Decision(Action.PUSH, "push"),
        timestamp=1.0,
        run_now=True,
    )
    for _ in range(100):
        result = advisor.poll_advice()
        if result is not None:
            break
        time.sleep(0.005)
    else:
        result = None
    advisor.close()

    assert result is not None
    assert result.request_id == 1


def test_generate_offline_review_writes_fallback_when_ollama_unavailable(
    monkeypatch, tmp_path
) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": 1.0,
                "frame_index": 1,
                "state": {"hp_pct": 0.9, "shield_pct": 0.9, "under_fire": False},
                "events": {"damage_delta": 0.0},
                "decision": {"reason": "No strong signal."},
                "arbiter": {"action": "NONE", "emitted": False},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def _raise(*args, **kwargs):
        raise error.URLError("offline")

    monkeypatch.setattr("apexcoach.llm_advisor.request.urlopen", _raise)
    out_path = tmp_path / "review.md"
    advisor = LlmAdvisor(
        LlmConfig(
            enabled=True,
            provider="ollama",
            offline_review_enabled=True,
            offline_review_output=str(out_path),
        )
    )
    generated = advisor.generate_offline_review(log_path, summary={"frames": 1})

    assert generated == str(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "LLM review failed" in text
    assert "llm_error" in text


def test_generate_offline_review_lmstudio_success(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": 2.0,
                "frame_index": 2,
                "state": {"hp_pct": 0.7, "shield_pct": 0.8, "under_fire": False},
                "events": {"damage_delta": 0.0},
                "decision": {"reason": "Enemy knock advantage with healthy team."},
                "arbiter": {"action": "PUSH", "emitted": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    expected_model_id = "qwen3.5-9b"

    def _urlopen(req, **kwargs):
        url = req.full_url
        if url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": expected_model_id}]})
        if url.endswith("/v1/chat/completions"):
            body = json.loads(req.data.decode("utf-8"))
            assert body["model"] == expected_model_id
            assert body["max_tokens"] == 1024
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "You converted a good push window.",
                                        "strengths": ["Fast follow-up on knock"],
                                        "improvements": ["Take cover before repeek"],
                                        "next_focus": ["Clean retreat timing after trade"],
                                        "timeline": [
                                            {"time_sec": 2.0, "advice": "Maintain pressure"}
                                        ],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("apexcoach.llm_advisor.request.urlopen", _urlopen)
    out_path = tmp_path / "review.md"
    advisor = LlmAdvisor(
        LlmConfig(
            enabled=True,
            provider="lmstudio",
            model="gpt-oss-swallow-20b",
            offline_review_model_name="qwen3.5-9b",
            base_url="http://127.0.0.1:1234",
            offline_review_enabled=True,
            offline_review_output=str(out_path),
            offline_review_language="en",
        )
    )
    generated = advisor.generate_offline_review(log_path, summary={"frames": 1})
    assert generated == str(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "## Summary" in text
    assert "You converted a good push window." in text


def test_generate_offline_review_enforces_japanese(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": 3.0,
                "frame_index": 3,
                "state": {"hp_pct": 0.5, "shield_pct": 0.6, "under_fire": True},
                "events": {"damage_delta": 0.2},
                "decision": {"reason": "High incoming damage in last 1s."},
                "arbiter": {"action": "RETREAT", "emitted": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    call_count = {"chat": 0}

    def _urlopen(req, **kwargs):
        url = req.full_url
        if url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": "qwen2.5-14b-instruct"}]})
        if url.endswith("/v1/chat/completions"):
            call_count["chat"] += 1
            if call_count["chat"] == 1:
                return FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "summary": "Retreat timing was mostly correct.",
                                            "strengths": ["Quick defensive reaction"],
                                            "improvements": ["Use cover before healing"],
                                            "next_focus": ["Damage reduction"],
                                            "timeline": [
                                                {"time_sec": 3.0, "advice": "Back to cover"}
                                            ],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "被弾時の退避判断は概ね適切です。",
                                        "strengths": ["防御反応が速い"],
                                        "improvements": ["回復前に遮蔽物を確保する"],
                                        "next_focus": ["被ダメージ抑制"],
                                        "timeline": [
                                            {"time_sec": 3.0, "advice": "遮蔽物へ戻る"}
                                        ],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("apexcoach.llm_advisor.request.urlopen", _urlopen)
    out_path = tmp_path / "review_ja.md"
    advisor = LlmAdvisor(
        LlmConfig(
            enabled=True,
            provider="lmstudio",
            base_url="http://127.0.0.1:1234",
            offline_review_enabled=True,
            offline_review_output=str(out_path),
            offline_review_language="ja",
        )
    )

    generated = advisor.generate_offline_review(log_path, summary={"frames": 1})
    assert generated == str(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "被弾時の退避判断" in text
