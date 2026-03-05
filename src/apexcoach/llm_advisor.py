from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error, request

from apexcoach.config import LlmConfig
from apexcoach.models import Action, ArbiterResult, Decision, GameState
from apexcoach.providers.factory import build_provider
from apexcoach.providers.types import LLMProviderConfig


class LlmAdvisor:
    """Optional low-frequency LLM helper for explanations and post-game review."""

    def __init__(self, config: LlmConfig) -> None:
        self.config = config
        self._last_advice_request_ts: float | None = None
        self._advice_in_flight = False
        self._advice_provider = None
        if self.config.enabled and self.config.advice_enabled:
            provider_cfg = LLMProviderConfig(
                base_url=self.config.base_url,
                model_name=self.config.model_name or self.config.model,
                timeout_ms=int(self.config.timeout_ms),
                temperature=float(self.config.temperature),
                max_tokens=int(self.config.llm_max_tokens),
                max_reason_chars=int(self.config.max_reason_chars),
                max_raw_response_chars=int(self.config.max_raw_response_chars),
                parse_retry_count=int(self.config.parse_retry_count),
                failure_threshold=int(self.config.failure_threshold),
                disable_seconds=float(self.config.disable_seconds),
                request_log_path=self.config.request_log_path,
                api_key=self.config.api_key,
            )
            try:
                self._advice_provider = build_provider(self.config.provider, provider_cfg)
            except ValueError:
                self._advice_provider = None

    def maybe_explain(
        self,
        state: GameState,
        decision: Decision,
        arbiter: ArbiterResult,
        run_now: bool,
    ) -> str | None:
        if (
            not self.config.enabled
            or not self.config.frame_reasoning_enabled
            or not run_now
        ):
            return None
        return (
            f"{arbiter.action.value}: {decision.reason} "
            f"(hp={state.hp_pct:.2f}, shield={state.shield_pct:.2f})"
        )

    def maybe_advise_decision(
        self,
        state: GameState,
        candidates: list[Decision],
        rule_decision: Decision,
        timestamp: float,
        run_now: bool,
    ) -> tuple[Decision | None, str | None]:
        if (
            not self.config.enabled
            or not self.config.advice_enabled
            or not run_now
            or self._advice_provider is None
        ):
            return None, None
        if self._advice_in_flight:
            return None, "llm_skip:in_flight"
        if not self._should_request(state=state, rule_decision=rule_decision):
            return None, None

        min_interval = max(0.0, float(self.config.min_request_interval_ms) / 1000.0)
        if (
            self._last_advice_request_ts is not None
            and timestamp - self._last_advice_request_ts < min_interval
        ):
            return None, "llm_skip:rate_limited"

        candidate_actions = [d.action for d in candidates] or [rule_decision.action]
        if rule_decision.action not in candidate_actions:
            candidate_actions = [rule_decision.action] + candidate_actions
        if all(a.value != "NONE" for a in candidate_actions):
            candidate_actions.append(Action.NONE)

        self._advice_in_flight = True
        self._last_advice_request_ts = timestamp
        try:
            result = self._advice_provider.generate_advice(
                state=state,
                candidate_actions=candidate_actions,
                context={"timestamp": timestamp},
            )
        finally:
            self._advice_in_flight = False

        if result is None:
            return None, "llm_none"

        selected = None
        for c in candidates:
            if c.action == result.action:
                selected = c
                break
        confidence = selected.confidence if selected is not None else 0.65
        decision = Decision(
            action=result.action,
            reason=result.reason,
            confidence=confidence,
            meta={
                "source": "llm",
                "provider": result.provider_name,
                "model": result.model_name,
                "request_id": result.request_id,
                "latency_ms": round(result.latency_ms, 1),
            },
        )
        reason = (
            f"{result.action.value}: {result.reason} "
            f"(provider={result.provider_name}, latency_ms={result.latency_ms:.1f})"
        )
        return decision, reason

    def generate_offline_review(
        self,
        session_log_path: str | Path,
        summary: dict[str, int],
    ) -> str | None:
        if not self.config.enabled or not self.config.offline_review_enabled:
            return None

        log_path = Path(session_log_path)
        if not log_path.exists():
            return None

        payload = _build_review_payload(
            log_path=log_path,
            max_events=self.config.offline_review_max_events,
            reason_max_chars=self.config.offline_review_reason_max_chars,
            summary=summary,
        )
        if payload is None:
            return None

        prompt = _build_review_prompt_with_budget(
            payload=payload,
            language=self.config.offline_review_language,
            max_chars=int(self.config.offline_review_prompt_max_chars),
        )
        result, provider_error = self._call_provider(prompt)
        result, language_note = self._ensure_output_language(
            result=result,
            target_language=self.config.offline_review_language,
        )
        provider_error = _merge_error(provider_error, language_note)
        rendered = _render_review_markdown(
            payload,
            result,
            provider_error=provider_error,
        )
        out_path = Path(self.config.offline_review_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        return str(out_path)

    def _should_request(self, state: GameState, rule_decision: Decision) -> bool:
        if state.ally_knock_recent or state.enemy_knock_recent:
            return True
        if state.recent_damage_1s >= 0.12:
            return True
        if rule_decision.action.value == "NONE":
            medium_risk = (
                state.under_fire
                or (state.hp_pct + state.shield_pct) <= 0.95
                or state.exposed_no_cover
                or state.low_ground_disadvantage
            )
            if medium_risk:
                return True
        return False

    def _call_provider(self, prompt: str) -> tuple[dict[str, Any] | None, str | None]:
        provider = (self.config.provider or "").strip().lower()
        if provider in {"ollama"}:
            return self._call_ollama(prompt)
        if provider in {"lmstudio", "lm_studio", "openai"}:
            return self._call_lmstudio(prompt)
        return None, f"Unsupported llm.provider: {self.config.provider!r}"

    def _call_ollama(self, prompt: str) -> tuple[dict[str, Any] | None, str | None]:
        body = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": float(self.config.temperature),
                "num_ctx": int(self.config.num_ctx),
            },
        }
        endpoint = self.config.base_url.rstrip("/") + "/api/generate"
        req = request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.config.timeout_seconds)) as resp:
                raw = resp.read().decode("utf-8")
            decoded = json.loads(raw)
            text = str(decoded.get("response", "")).strip()
            parsed = _parse_json_object_text(text)
            if parsed is None:
                return None, "Ollama response was not valid JSON object text."
            return parsed, None
        except error.HTTPError as exc:
            return None, f"Ollama HTTP {exc.code}: {_read_http_error_body(exc)}"
        except (json.JSONDecodeError, error.URLError, TimeoutError, OSError):
            return None, "Ollama API request failed."

    def _call_lmstudio(self, prompt: str) -> tuple[dict[str, Any] | None, str | None]:
        model_id, model_note = self._resolve_lmstudio_model_id()
        base_body = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an Apex Legends coaching assistant. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.llm_max_tokens),
        }
        response_format = (self.config.lmstudio_response_format or "").strip().lower()
        format_body = dict(base_body)
        format_label = "formatted mode"
        if response_format == "json_schema":
            format_body["response_format"] = _lmstudio_json_schema_response_format()
            format_label = "json_schema mode"
        elif response_format == "text":
            format_body["response_format"] = {"type": "text"}
            format_label = "text mode"
        else:
            format_body["response_format"] = _lmstudio_json_schema_response_format()
            format_label = "json_schema mode"

        attempts: list[tuple[dict[str, Any], str]] = [
            (format_body, format_label),
            (base_body, "plain mode"),
        ]

        last_error: str | None = model_note
        for body, label in attempts:
            parsed, err = self._call_lmstudio_chat(body)
            if parsed is not None:
                return parsed, model_note
            last_error = _merge_error(last_error, f"{label}: {err}")
        return None, last_error or "LM Studio API request failed."

    def _call_lmstudio_chat(
        self,
        body: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        endpoint = _build_lmstudio_endpoint(self.config.base_url, "/chat/completions")
        req = request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=self._build_api_headers(),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=float(self.config.timeout_seconds)) as resp:
                raw = resp.read().decode("utf-8")
            decoded = json.loads(raw)
            choices = decoded.get("choices")
            if not isinstance(choices, list) or not choices:
                return None, "LM Studio response missing choices."
            first = choices[0]
            if not isinstance(first, dict):
                return None, "LM Studio choices[0] is not an object."
            message = first.get("message", {})
            if not isinstance(message, dict):
                return None, "LM Studio response message is not an object."
            content = _extract_chat_content(message.get("content"))
            parsed = _parse_json_object_text(content)
            if parsed is None:
                return None, "LM Studio message content was not valid JSON object text."
            return parsed, None
        except error.HTTPError as exc:
            return None, f"LM Studio HTTP {exc.code}: {_read_http_error_body(exc)}"
        except (
            json.JSONDecodeError,
            error.URLError,
            TimeoutError,
            OSError,
            TypeError,
            ValueError,
        ):
            return None, "LM Studio API request failed."

    def _resolve_lmstudio_model_id(self) -> tuple[str, str | None]:
        configured = (self.config.model or "").strip()
        model_ids, err = self._fetch_lmstudio_model_ids()
        if not model_ids:
            return configured, err

        if configured in model_ids:
            return configured, None

        normalized_cfg = _normalize_model_name(configured)
        for model_id in model_ids:
            normalized_id = _normalize_model_name(model_id)
            if normalized_cfg and (
                normalized_cfg in normalized_id or normalized_id in normalized_cfg
            ):
                return model_id, f"LM Studio model auto-selected: {model_id}"

        return model_ids[0], f"LM Studio model auto-selected: {model_ids[0]}"

    def _fetch_lmstudio_model_ids(self) -> tuple[list[str], str | None]:
        endpoint = _build_lmstudio_endpoint(self.config.base_url, "/models")
        req = request.Request(endpoint, headers=self._build_api_headers(), method="GET")
        try:
            with request.urlopen(req, timeout=float(self.config.timeout_seconds)) as resp:
                raw = resp.read().decode("utf-8")
            decoded = json.loads(raw)
            data = decoded.get("data")
            if not isinstance(data, list):
                return [], "LM Studio /v1/models returned unexpected payload."
            ids: list[str] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    ids.append(model_id.strip())
            return ids, None
        except error.HTTPError as exc:
            return [], f"LM Studio model list HTTP {exc.code}: {_read_http_error_body(exc)}"
        except (
            json.JSONDecodeError,
            error.URLError,
            TimeoutError,
            OSError,
            TypeError,
            ValueError,
        ):
            return [], "LM Studio model list request failed."

    def _build_api_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = (self.config.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _ensure_output_language(
        self,
        result: dict[str, Any] | None,
        target_language: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if result is None:
            return None, None
        lang = (target_language or "").strip().lower()
        if lang not in {"ja", "japanese", "日本語"}:
            return result, None
        if _contains_japanese_text_in_json(result):
            return result, None

        prompt = _build_translation_prompt(result, target_language="ja")
        translated, err = self._call_provider(prompt)
        if translated and _contains_japanese_text_in_json(translated):
            return translated, "LLM output translated to Japanese."
        return result, _merge_error("Japanese enforcement failed", err)


def _build_review_payload(
    log_path: Path,
    max_events: int,
    reason_max_chars: int,
    summary: dict[str, int],
) -> dict[str, Any] | None:
    total_frames = 0
    first_ts: float | None = None
    last_ts: float | None = None
    under_fire_frames = 0
    low_resource_frames = 0
    questionable_emits = 0
    event_rows: list[dict[str, Any]] = []

    limit = max(1, int(max_events))

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue

            total_frames += 1
            ts = float(record.get("timestamp", 0.0))
            if first_ts is None:
                first_ts = ts
            last_ts = ts

            state = record.get("state") or {}
            arbiter = record.get("arbiter") or {}
            decision = record.get("decision") or {}

            hp = float(state.get("hp_pct", 1.0))
            shield = float(state.get("shield_pct", 1.0))
            total = hp + shield

            if bool(state.get("under_fire", False)):
                under_fire_frames += 1
            if total <= 0.65:
                low_resource_frames += 1

            emitted = bool(arbiter.get("emitted", False))
            action = str(arbiter.get("action", "NONE"))
            if (
                emitted
                and action in ("HEAL", "RETREAT")
                and total >= 1.6
                and not bool(state.get("under_fire", False))
            ):
                questionable_emits += 1

            if emitted and action != "NONE" and len(event_rows) < limit:
                event_rows.append(
                    {
                        "t": round(ts, 2),
                        "action": action,
                        "reason": _truncate_text(
                            str(decision.get("reason", "")),
                            max_chars=max(24, int(reason_max_chars)),
                        ),
                        "hp": round(hp, 3),
                        "shield": round(shield, 3),
                        "under_fire": bool(state.get("under_fire", False)),
                        "allies_alive": int(state.get("allies_alive", 3)),
                        "allies_down": int(state.get("allies_down", 0)),
                    }
                )

    if total_frames == 0 or first_ts is None or last_ts is None:
        return None

    duration = max(0.0, last_ts - first_ts)
    return {
        "meta": {
            "frames": total_frames,
            "duration_sec": round(duration, 2),
            "source_log": str(log_path),
        },
        "summary": summary,
        "ratios": {
            "under_fire_ratio": round(under_fire_frames / total_frames, 4),
            "low_resource_ratio": round(low_resource_frames / total_frames, 4),
        },
        "quality": {
            "questionable_emit_count": questionable_emits,
        },
        "timeline": event_rows,
    }


def _build_review_prompt(payload: dict[str, Any], language: str) -> str:
    lang = language.strip() or "ja"
    input_json = json.dumps(payload, ensure_ascii=False)
    return (
        "You are an Apex Legends coach.\n"
        f"Respond in {lang}.\n"
        "If the target language is Japanese, every string field must be natural Japanese.\n"
        "Do not output English unless it is a proper noun.\n"
        "Use only the provided JSON data. Do not invent events.\n"
        "Return JSON only with this schema:\n"
        "{"
        '"summary":"string",'
        '"strengths":["string","..."],'
        '"improvements":["string","..."],'
        '"next_focus":["string","..."],'
        '"timeline":[{"time_sec":number,"advice":"string"}]'
        "}\n"
        "Keep each bullet concise and actionable.\n"
        f"INPUT_JSON={input_json}\n"
    )


def _build_translation_prompt(
    llm_json: dict[str, Any],
    target_language: str,
) -> str:
    source = json.dumps(llm_json, ensure_ascii=False)
    return (
        "Translate all user-facing string values in the JSON to the target language.\n"
        f"Target language: {target_language}\n"
        "Keep the JSON structure and numeric values unchanged.\n"
        "Return JSON only.\n"
        f"INPUT_JSON={source}\n"
    )


def _build_review_prompt_with_budget(
    payload: dict[str, Any],
    language: str,
    max_chars: int,
) -> str:
    budget = max(2000, int(max_chars))
    working = json.loads(json.dumps(payload, ensure_ascii=False))
    prompt = _build_review_prompt(working, language=language)
    timeline = working.get("timeline")
    if not isinstance(timeline, list):
        return prompt

    while len(prompt) > budget and len(timeline) > 6:
        timeline.pop()
        prompt = _build_review_prompt(working, language=language)

    if len(prompt) > budget and timeline:
        working["timeline"] = timeline[:3]
        prompt = _build_review_prompt(working, language=language)

    if len(prompt) > budget:
        working["timeline"] = []
        prompt = _build_review_prompt(working, language=language)

    return prompt


def _render_review_markdown(
    payload: dict[str, Any],
    llm_json: dict[str, Any] | None,
    provider_error: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# ApexCoach Local LLM Review")
    lines.append("")
    meta = payload.get("meta", {})
    lines.append(f"- frames: {meta.get('frames', 0)}")
    lines.append(f"- duration_sec: {meta.get('duration_sec', 0)}")
    lines.append(f"- source_log: {meta.get('source_log', '')}")
    lines.append("")

    if not llm_json:
        lines.append("## Summary")
        lines.append("LLM review failed or returned invalid JSON.")
        lines.append("")
        lines.append("## Fallback Notes")
        q = payload.get("quality", {}).get("questionable_emit_count", 0)
        under_fire_ratio = payload.get("ratios", {}).get("under_fire_ratio", 0)
        lines.append(f"- questionable_emit_count: {q}")
        lines.append(f"- under_fire_ratio: {under_fire_ratio}")
        if provider_error:
            lines.append(f"- llm_error: {provider_error}")
        return "\n".join(lines) + "\n"

    lines.append("## Summary")
    lines.append(str(llm_json.get("summary", "")).strip())
    lines.append("")

    for title, key in (
        ("Strengths", "strengths"),
        ("Improvements", "improvements"),
        ("Next Focus", "next_focus"),
    ):
        lines.append(f"## {title}")
        items = llm_json.get(key, [])
        if isinstance(items, list) and items:
            for item in items:
                lines.append(f"- {str(item)}")
        else:
            lines.append("- N/A")
        lines.append("")

    lines.append("## Timeline Advice")
    timeline = llm_json.get("timeline", [])
    if isinstance(timeline, list) and timeline:
        for item in timeline:
            if not isinstance(item, dict):
                continue
            t = item.get("time_sec", "")
            advice = item.get("advice", "")
            lines.append(f"- {t}s: {advice}")
    else:
        lines.append("- N/A")
    lines.append("")
    return "\n".join(lines)


def _parse_json_object_text(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate.startswith("{"):
                continue
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if 0 <= first_brace < last_brace:
        candidate = raw[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _lmstudio_json_schema_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "apexcoach_review",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "improvements": {"type": "array", "items": {"type": "string"}},
                    "next_focus": {"type": "array", "items": {"type": "string"}},
                    "timeline": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "time_sec": {"type": "number"},
                                "advice": {"type": "string"},
                            },
                            "required": ["time_sec", "advice"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "summary",
                    "strengths",
                    "improvements",
                    "next_focus",
                    "timeline",
                ],
                "additionalProperties": False,
            },
        },
    }


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 3:
        return text[:max_chars]
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _extract_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks)
    return ""


def _contains_japanese_text_in_json(value: Any) -> bool:
    for text in _iter_string_values(value):
        if _contains_japanese_char(text):
            return True
    return False


def _iter_string_values(value: Any):
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _iter_string_values(v)
        return
    if isinstance(value, list):
        for v in value:
            yield from _iter_string_values(v)


def _contains_japanese_char(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:
            return True
        if 0x4E00 <= code <= 0x9FFF:
            return True
        if 0x3400 <= code <= 0x4DBF:
            return True
    return False


def _normalize_model_name(name: str) -> str:
    normalized = (name or "").strip().lower()
    if normalized.endswith(".gguf"):
        normalized = normalized[:-5]
    return normalized


def _merge_error(existing: str | None, extra: str | None) -> str | None:
    if not extra:
        return existing
    if not existing:
        return extra
    return f"{existing}; {extra}"


def _read_http_error_body(exc: error.HTTPError) -> str:
    try:
        body = exc.read()
    except OSError:
        return ""
    if not body:
        return ""
    text = body.decode("utf-8", errors="replace").strip()
    if len(text) > 300:
        return text[:300] + "..."
    return text


def _build_lmstudio_endpoint(base_url: str, path: str) -> str:
    base = (base_url or "").rstrip("/")
    clean_path = path if path.startswith("/") else f"/{path}"
    if base.endswith("/v1"):
        return f"{base}{clean_path}"
    return f"{base}/v1{clean_path}"
