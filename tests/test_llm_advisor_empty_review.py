import json

from apexcoach.config import LlmConfig
from apexcoach.llm_advisor import LlmAdvisor


def test_generate_offline_review_rejects_empty_json_content(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": 4.0,
                "frame_index": 4,
                "state": {"hp_pct": 0.5, "shield_pct": 0.5, "under_fire": True},
                "events": {"damage_delta": 0.1},
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

    def _urlopen(req, **kwargs):
        url = req.full_url
        if url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": "gpt-oss-swallow-20b"}]})
        if url.endswith("/v1/chat/completions"):
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
    out_path = tmp_path / "review_empty.md"
    advisor = LlmAdvisor(
        LlmConfig(
            enabled=True,
            provider="lmstudio",
            base_url="http://127.0.0.1:1234",
            offline_review_enabled=True,
            offline_review_output=str(out_path),
            offline_review_language="en",
        )
    )

    generated = advisor.generate_offline_review(log_path, summary={"frames": 1})
    assert generated == str(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "LLM review failed or returned invalid JSON." in text
    assert "LLM returned empty review content." in text
