import json
from urllib import error

from apexcoach.config import LlmConfig
from apexcoach.llm_advisor import LlmAdvisor, _build_review_payload
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

    expected_model_id = "qwen2.5-14b-instruct-q4_k_m"

    def _urlopen(req, **kwargs):
        url = req.full_url
        if url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": expected_model_id}]})
        if url.endswith("/v1/chat/completions"):
            body = json.loads(req.data.decode("utf-8"))
            assert body["model"] == expected_model_id
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
            model="qwen2.5-14b-instruct-q4_k_m.gguf",
            base_url="http://127.0.0.1:1234",
            offline_review_enabled=True,
            offline_review_output=str(out_path),
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
