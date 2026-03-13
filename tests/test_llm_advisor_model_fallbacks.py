import json

from apexcoach.config import LlmConfig
from apexcoach.llm_advisor import LlmAdvisor
from apexcoach.models import Action, Decision, GameState
from apexcoach.providers.types import LLMAdviceResult


def test_realtime_advice_tries_multiple_models(monkeypatch) -> None:
    class FakeProvider:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate_advice(self, state, candidate_actions, context=None):
            if self.model_name == "gpt-oss-swallow-20b":
                return None
            return LLMAdviceResult(
                action=Action.RETREAT,
                reason="fallback advice",
                request_id="req-1",
                provider_name="mock",
                model_name=self.model_name,
                latency_ms=10.0,
            )

    def _build_provider(provider_name, config):
        return FakeProvider(config.model_name)

    monkeypatch.setattr("apexcoach.llm_advisor.build_provider", _build_provider)
    advisor = LlmAdvisor(
        LlmConfig(
            enabled=True,
            provider="lmstudio",
            model_names=["gpt-oss-swallow-20b", "qwen3.5-9b"],
        )
    )
    state = GameState(timestamp=1.0, hp_pct=0.3, shield_pct=0.2, under_fire=True, recent_damage_1s=0.2)
    decision, reason = advisor.maybe_advise_decision(
        state=state,
        candidates=[
            Decision(action=Action.RETREAT, reason="High incoming damage in last 1s.", confidence=0.9),
            Decision(action=Action.NONE, reason="No strong signal.", confidence=0.5),
        ],
        rule_decision=Decision(action=Action.RETREAT, reason="High incoming damage in last 1s.", confidence=0.9),
        timestamp=1.0,
        run_now=True,
    )

    assert decision is not None
    assert decision.action == Action.RETREAT
    assert decision.meta["model"] == "qwen3.5-9b"
    assert reason is not None and "fallback advice" in reason


def test_offline_review_tries_multiple_models(monkeypatch, tmp_path) -> None:
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

    def _urlopen(req, **kwargs):
        url = req.full_url
        if url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": "gpt-oss-swallow-20b"}, {"id": "qwen3.5-9b"}]})
        if url.endswith("/v1/chat/completions"):
            body = json.loads(req.data.decode("utf-8"))
            if body["model"] == "gpt-oss-swallow-20b":
                return FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "summary": "",
                                            "strengths": [],
                                            "improvements": [],
                                            "next_focus": [],
                                            "timeline": [{"time_sec": 0, "advice": ""}],
                                        }
                                    )
                                },
                                "finish_reason": "stop",
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
                                        "summary": "You converted a good push window.",
                                        "strengths": ["Fast follow-up on knock"],
                                        "improvements": ["Take cover before repeek"],
                                        "next_focus": ["Clean retreat timing after trade"],
                                        "timeline": [{"time_sec": 2.0, "advice": "Maintain pressure"}],
                                    }
                                )
                            },
                            "finish_reason": "stop",
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
            model_names=["gpt-oss-swallow-20b"],
            offline_review_model_names=["gpt-oss-swallow-20b", "qwen3.5-9b"],
            offline_review_enabled=True,
            offline_review_output=str(out_path),
            offline_review_language="en",
        )
    )

    generated = advisor.generate_offline_review(log_path, summary={"frames": 1})
    assert generated == str(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "You converted a good push window." in text


def test_realtime_advice_uses_loaded_models_when_no_list_configured(monkeypatch) -> None:
    built_models = []

    class FakeProvider:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate_advice(self, state, candidate_actions, context=None):
            return None

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def _build_provider(provider_name, config):
        built_models.append(config.model_name)
        return FakeProvider(config.model_name)

    def _urlopen(req, **kwargs):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": "model-a"}, {"id": "model-b"}]})
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("apexcoach.llm_advisor.build_provider", _build_provider)
    monkeypatch.setattr("apexcoach.llm_advisor.request.urlopen", _urlopen)

    _ = LlmAdvisor(
        LlmConfig(
            enabled=True,
            provider="lmstudio",
            model_name="",
            model_names=[],
        )
    )

    assert built_models == ["model-a", "model-b"]
