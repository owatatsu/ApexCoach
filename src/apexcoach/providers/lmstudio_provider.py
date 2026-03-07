from __future__ import annotations

import json
import socket
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib import error, request

from apexcoach.models import Action, GameState
from apexcoach.providers.base import BaseLLMProvider
from apexcoach.providers.prompt_builder import build_messages
from apexcoach.providers.response_parser import ParseErrorType, parse_advice_text
from apexcoach.providers.types import LLMAdviceResult, LLMProviderConfig
from apexcoach.utils.jsonl_logger import JsonlLogger


class LMStudioProvider(BaseLLMProvider):
    def __init__(self, config: LLMProviderConfig) -> None:
        super().__init__(config)
        self._base_url = _normalize_base_url(config.base_url)
        self._consecutive_failures = 0
        self._disabled_until_monotonic = 0.0
        self._last_disabled_log_at = 0.0
        self._logger = JsonlLogger(config.request_log_path, enabled=True)

    def name(self) -> str:
        return "lmstudio"

    def healthcheck(self) -> bool:
        endpoint = self._endpoint("/models")
        req = request.Request(
            endpoint,
            headers=self._headers(),
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=max(0.05, self.config.timeout_ms / 1000.0)) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            return isinstance(payload.get("data"), list)
        except (error.URLError, error.HTTPError, json.JSONDecodeError, TimeoutError, OSError):
            return False

    def generate_advice(
        self,
        state: GameState,
        candidate_actions: Sequence[Action],
        context: dict[str, Any] | None = None,
    ) -> LLMAdviceResult | None:
        now_mono = time.monotonic()
        if now_mono < self._disabled_until_monotonic:
            # Avoid log spam while circuit breaker is active.
            if now_mono - self._last_disabled_log_at >= 1.0:
                self._last_disabled_log_at = now_mono
                self._log_request(
                    request_id="disabled",
                    state=state,
                    candidate_actions=list(candidate_actions),
                    latency_ms=0.0,
                    timed_out=False,
                    raw_response="",
                    parse_ok=False,
                    validated_ok=False,
                    result_action=None,
                    result_reason=None,
                    error_type="provider_disabled",
                )
            return None

        candidates = list(candidate_actions)
        if not candidates:
            return None

        request_id = str(uuid.uuid4())
        messages = build_messages(
            state=state,
            candidate_actions=candidates,
            max_reason_chars=self.config.max_reason_chars,
            language="ja",
            context=context,
        )
        retry_count = max(0, int(self.config.parse_retry_count))

        last_parse_error: ParseErrorType | None = None
        last_raw_response = ""
        for attempt in range(retry_count + 1):
            started = time.perf_counter()
            raw_text, transport_error, timed_out = self._request_chat(messages)
            latency_ms = (time.perf_counter() - started) * 1000.0
            last_raw_response = raw_text

            if transport_error is not None:
                self._record_failure(
                    hard=timed_out or transport_error.startswith("network_error")
                )
                self._log_request(
                    request_id=request_id,
                    state=state,
                    candidate_actions=candidates,
                    latency_ms=latency_ms,
                    timed_out=timed_out,
                    raw_response=raw_text,
                    parse_ok=False,
                    validated_ok=False,
                    result_action=None,
                    result_reason=None,
                    error_type=transport_error,
                )
                # timeout/network errors are not retried
                return None

            parsed = parse_advice_text(
                raw_text=raw_text,
                candidate_actions=candidates,
                max_reason_chars=self.config.max_reason_chars,
            )
            last_parse_error = parsed.error_type
            if parsed.validated_ok and parsed.action is not None and parsed.reason is not None:
                self._consecutive_failures = 0
                result = LLMAdviceResult(
                    action=parsed.action,
                    reason=parsed.reason,
                    request_id=request_id,
                    provider_name=self.name(),
                    model_name=self.config.model_name,
                    latency_ms=latency_ms,
                    raw_response=_truncate(raw_text, self.config.max_raw_response_chars),
                )
                self._log_request(
                    request_id=request_id,
                    state=state,
                    candidate_actions=candidates,
                    latency_ms=latency_ms,
                    timed_out=False,
                    raw_response=raw_text,
                    parse_ok=parsed.parse_ok,
                    validated_ok=parsed.validated_ok,
                    result_action=result.action.value,
                    result_reason=result.reason,
                    error_type=None,
                )
                return result

            self._log_request(
                request_id=request_id,
                state=state,
                candidate_actions=candidates,
                latency_ms=latency_ms,
                timed_out=False,
                raw_response=raw_text,
                parse_ok=parsed.parse_ok,
                validated_ok=parsed.validated_ok,
                result_action=None,
                result_reason=None,
                error_type=parsed.error_type.value if parsed.error_type else "parse_failed",
            )
            if attempt >= retry_count:
                break

        self._record_failure()
        if last_parse_error is not None:
            self._log_request(
                request_id=request_id,
                state=state,
                candidate_actions=candidates,
                latency_ms=0.0,
                timed_out=False,
                raw_response=last_raw_response,
                parse_ok=False,
                validated_ok=False,
                result_action=None,
                result_reason=None,
                error_type=f"final_parse_error:{last_parse_error.value}",
            )
        return None

    def _request_chat(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, str | None, bool]:
        body = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_tokens),
            "response_format": {"type": "text"},
        }
        req = request.Request(
            self._endpoint("/chat/completions"),
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=max(0.05, self.config.timeout_ms / 1000.0)) as resp:
                raw = resp.read().decode("utf-8")
            decoded = json.loads(raw)
            choice_list = decoded.get("choices")
            if not isinstance(choice_list, list) or not choice_list:
                return "", "missing_choices", False
            first = choice_list[0]
            if not isinstance(first, dict):
                return "", "invalid_choice", False
            message = first.get("message")
            if not isinstance(message, dict):
                return "", "invalid_message", False
            content = message.get("content", "")
            if isinstance(content, str):
                return content, None, False
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                return "\n".join(parts), None, False
            return str(content), None, False
        except error.HTTPError as exc:
            body = _read_http_error(exc)
            return "", f"http_{exc.code}:{body}", False
        except TimeoutError:
            return "", "timeout", True
        except error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, TimeoutError) or isinstance(reason, socket.timeout):
                return "", "timeout", True
            return "", f"network_error:{type(reason).__name__}", False
        except (json.JSONDecodeError, OSError, ValueError):
            return "", "invalid_provider_response", False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = (self.config.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _endpoint(self, path: str) -> str:
        base = self._base_url.rstrip("/")
        clean = path if path.startswith("/") else f"/{path}"
        return f"{base}{clean}"

    def _record_failure(self, hard: bool = False) -> None:
        if hard:
            self._disabled_until_monotonic = time.monotonic() + float(self.config.disable_seconds)
            self._consecutive_failures = 0
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= int(self.config.failure_threshold):
            self._disabled_until_monotonic = time.monotonic() + float(self.config.disable_seconds)
            self._consecutive_failures = 0

    def _log_request(
        self,
        request_id: str,
        state: GameState,
        candidate_actions: list[Action],
        latency_ms: float,
        timed_out: bool,
        raw_response: str,
        parse_ok: bool,
        validated_ok: bool,
        result_action: str | None,
        result_reason: str | None,
        error_type: str | None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "provider": self.name(),
            "model_name": self.config.model_name,
            "candidate_actions": [a.value for a in candidate_actions],
            "state_summary": {
                "hp_pct": round(state.hp_pct, 3),
                "shield_pct": round(state.shield_pct, 3),
                "allies_alive": state.allies_alive,
                "allies_down": state.allies_down,
                "under_fire": state.under_fire,
                "recent_damage_1s": round(state.recent_damage_1s, 3),
            },
            "latency_ms": round(latency_ms, 2),
            "timeout_ms": int(self.config.timeout_ms),
            "timed_out": timed_out,
            "raw_response": _truncate(raw_response, self.config.max_raw_response_chars),
            "parse_ok": parse_ok,
            "validated_ok": validated_ok,
            "result_action": result_action,
            "result_reason": result_reason,
            "error_type": error_type,
        }
        self._logger.write(record)


def _read_http_error(exc: error.HTTPError) -> str:
    try:
        raw = exc.read()
    except OSError:
        return ""
    if not raw:
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    return _truncate(text, 240)


def _truncate(text: str, max_chars: int) -> str:
    s = text or ""
    limit = max(16, int(max_chars))
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _normalize_base_url(base_url: str) -> str:
    raw = (base_url or "").strip()
    if not raw:
        return "http://127.0.0.1:1234/v1"
    parts = urlsplit(raw)
    host = parts.hostname or ""
    if host.lower() != "localhost":
        return raw
    replaced_netloc = parts.netloc.replace("localhost", "127.0.0.1")
    return urlunsplit((parts.scheme, replaced_netloc, parts.path, parts.query, parts.fragment))
