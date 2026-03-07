import json
import socket
from urllib import error

from apexcoach.models import Action, GameState
from apexcoach.providers.lmstudio_provider import LMStudioProvider
from apexcoach.providers.types import LLMProviderConfig


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _state() -> GameState:
    return GameState(
        timestamp=1.0,
        hp_pct=0.5,
        shield_pct=0.2,
        allies_alive=2,
        allies_down=1,
        recent_damage_1s=0.2,
        under_fire=True,
    )


def test_lmstudio_provider_success(tmp_path, monkeypatch) -> None:
    cfg = LLMProviderConfig(
        base_url="http://localhost:1234/v1",
        model_name="qwen3.5-9b-instruct",
        request_log_path=str(tmp_path / "llm_req.jsonl"),
    )
    provider = LMStudioProvider(cfg)

    seen = {"body": None}

    def _urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"RETREAT","reason":"被弾中のため下がる"}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("apexcoach.providers.lmstudio_provider.request.urlopen", _urlopen)
    result = provider.generate_advice(
        state=_state(),
        candidate_actions=[Action.RETREAT, Action.NONE],
        context={
            "timestamp": 1.0,
            "decision_trigger": "damage_spike",
            "rule_decision": {
                "action": "RETREAT",
                "reason": "High incoming damage in last 1s.",
                "confidence": 0.9,
            },
            "candidate_details": [
                {
                    "action": "RETREAT",
                    "reason": "High incoming damage in last 1s.",
                    "confidence": 0.9,
                },
                {
                    "action": "NONE",
                    "reason": "No strong signal.",
                    "confidence": 0.5,
                },
            ],
        },
    )
    assert result is not None
    assert result.action == Action.RETREAT
    assert seen["body"] is not None
    messages = seen["body"]["messages"]
    user_payload = json.loads(messages[1]["content"])
    assert user_payload["state"]["total_hp_shield"] == 0.7
    assert user_payload["state"]["vitals_confidence"] == 1.0
    assert user_payload["rule_decision"]["action"] == "RETREAT"
    assert user_payload["candidate_details"][0]["confidence"] == 0.9
    assert user_payload["request_context"]["decision_trigger"] == "damage_spike"


def test_lmstudio_provider_timeout_returns_none(tmp_path, monkeypatch) -> None:
    cfg = LLMProviderConfig(
        base_url="http://localhost:1234/v1",
        model_name="qwen3.5-9b-instruct",
        request_log_path=str(tmp_path / "llm_req.jsonl"),
    )
    provider = LMStudioProvider(cfg)

    def _urlopen(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("apexcoach.providers.lmstudio_provider.request.urlopen", _urlopen)
    result = provider.generate_advice(
        state=_state(),
        candidate_actions=[Action.RETREAT, Action.NONE],
    )
    assert result is None


def test_lmstudio_provider_connection_failure_returns_none(tmp_path, monkeypatch) -> None:
    cfg = LLMProviderConfig(
        base_url="http://localhost:1234/v1",
        model_name="qwen3.5-9b-instruct",
        request_log_path=str(tmp_path / "llm_req.jsonl"),
    )
    provider = LMStudioProvider(cfg)

    def _urlopen(req, timeout=None):
        raise error.URLError("offline")

    monkeypatch.setattr("apexcoach.providers.lmstudio_provider.request.urlopen", _urlopen)
    result = provider.generate_advice(
        state=_state(),
        candidate_actions=[Action.RETREAT, Action.NONE],
    )
    assert result is None


def test_lmstudio_provider_invalid_json_retries_once(tmp_path, monkeypatch) -> None:
    cfg = LLMProviderConfig(
        base_url="http://localhost:1234/v1",
        model_name="qwen3.5-9b-instruct",
        parse_retry_count=1,
        request_log_path=str(tmp_path / "llm_req.jsonl"),
    )
    provider = LMStudioProvider(cfg)
    calls = {"count": 0}

    def _urlopen(req, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeResponse(
                {"choices": [{"message": {"content": "not json at all"}}]}
            )
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"NONE","reason":"不確実なので維持"}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("apexcoach.providers.lmstudio_provider.request.urlopen", _urlopen)
    result = provider.generate_advice(
        state=_state(),
        candidate_actions=[Action.NONE, Action.RETREAT],
    )
    assert calls["count"] == 2
    assert result is not None
    assert result.action == Action.NONE


def test_lmstudio_provider_writes_jsonl_log(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "llm_req.jsonl"
    cfg = LLMProviderConfig(
        base_url="http://localhost:1234/v1",
        model_name="qwen3.5-9b-instruct",
        request_log_path=str(log_path),
    )
    provider = LMStudioProvider(cfg)

    def _urlopen(req, timeout=None):
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"RETREAT","reason":"被弾中のため下がる"}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("apexcoach.providers.lmstudio_provider.request.urlopen", _urlopen)
    _ = provider.generate_advice(
        state=_state(),
        candidate_actions=[Action.RETREAT, Action.NONE],
    )
    text = log_path.read_text(encoding="utf-8")
    assert "request_id" in text
    assert "result_action" in text


def test_lmstudio_provider_normalizes_localhost_base_url(tmp_path, monkeypatch) -> None:
    cfg = LLMProviderConfig(
        base_url="http://localhost:1234/v1",
        model_name="qwen3.5-9b-instruct",
        request_log_path=str(tmp_path / "llm_req.jsonl"),
    )
    provider = LMStudioProvider(cfg)
    seen = {"url": ""}

    def _urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"NONE","reason":"不確実なので維持"}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("apexcoach.providers.lmstudio_provider.request.urlopen", _urlopen)
    _ = provider.generate_advice(
        state=_state(),
        candidate_actions=[Action.NONE, Action.RETREAT],
    )
    assert "127.0.0.1" in seen["url"]


def test_lmstudio_provider_timeout_reason_inside_urlerror(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "llm_req.jsonl"
    cfg = LLMProviderConfig(
        base_url="http://127.0.0.1:1234/v1",
        model_name="qwen3.5-9b-instruct",
        request_log_path=str(log_path),
    )
    provider = LMStudioProvider(cfg)

    def _urlopen(req, timeout=None):
        raise error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr("apexcoach.providers.lmstudio_provider.request.urlopen", _urlopen)
    result = provider.generate_advice(
        state=_state(),
        candidate_actions=[Action.NONE, Action.RETREAT],
    )
    assert result is None
    text = log_path.read_text(encoding="utf-8")
    assert '"timed_out": true' in text
    assert '"error_type": "timeout"' in text
