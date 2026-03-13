from __future__ import annotations

from dataclasses import dataclass

from apexcoach.models import Action


@dataclass(slots=True)
class LLMProviderConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    model_name: str = ""
    timeout_ms: int = 300
    temperature: float = 0.1
    max_tokens: int = 64
    max_reason_chars: int = 48
    max_raw_response_chars: int = 1200
    parse_retry_count: int = 1
    failure_threshold: int = 5
    disable_seconds: float = 10.0
    request_log_path: str = "logs/llm_requests.jsonl"
    api_key: str = "lm-studio"


@dataclass(slots=True)
class LLMAdviceResult:
    action: Action
    reason: str
    request_id: str
    provider_name: str
    model_name: str
    latency_ms: float
    raw_response: str | None = None
