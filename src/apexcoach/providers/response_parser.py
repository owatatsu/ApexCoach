from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from apexcoach.models import Action


class ParseErrorType(str, Enum):
    INVALID_JSON = "invalid_json"
    NOT_OBJECT = "not_object"
    MISSING_ACTION = "missing_action"
    INVALID_ACTION = "invalid_action"
    ACTION_NOT_CANDIDATE = "action_not_candidate"
    MISSING_REASON = "missing_reason"
    EMPTY_REASON = "empty_reason"


@dataclass(slots=True)
class ParseAdviceResult:
    action: Action | None
    reason: str | None
    parse_ok: bool
    validated_ok: bool
    error_type: ParseErrorType | None


def parse_advice_text(
    raw_text: str,
    candidate_actions: list[Action],
    max_reason_chars: int,
) -> ParseAdviceResult:
    obj = _parse_json_object(raw_text)
    if obj is None:
        return ParseAdviceResult(
            action=None,
            reason=None,
            parse_ok=False,
            validated_ok=False,
            error_type=ParseErrorType.INVALID_JSON,
        )
    if not isinstance(obj, dict):
        return ParseAdviceResult(
            action=None,
            reason=None,
            parse_ok=True,
            validated_ok=False,
            error_type=ParseErrorType.NOT_OBJECT,
        )

    raw_action = obj.get("action")
    if not isinstance(raw_action, str):
        return ParseAdviceResult(
            action=None,
            reason=None,
            parse_ok=True,
            validated_ok=False,
            error_type=ParseErrorType.MISSING_ACTION,
        )
    action_value = raw_action.strip().upper()
    try:
        action = Action(action_value)
    except ValueError:
        return ParseAdviceResult(
            action=None,
            reason=None,
            parse_ok=True,
            validated_ok=False,
            error_type=ParseErrorType.INVALID_ACTION,
        )
    if action not in candidate_actions:
        return ParseAdviceResult(
            action=action,
            reason=None,
            parse_ok=True,
            validated_ok=False,
            error_type=ParseErrorType.ACTION_NOT_CANDIDATE,
        )

    raw_reason = obj.get("reason")
    if not isinstance(raw_reason, str):
        return ParseAdviceResult(
            action=action,
            reason=None,
            parse_ok=True,
            validated_ok=False,
            error_type=ParseErrorType.MISSING_REASON,
        )
    reason = raw_reason.strip()
    if not reason:
        return ParseAdviceResult(
            action=action,
            reason=None,
            parse_ok=True,
            validated_ok=False,
            error_type=ParseErrorType.EMPTY_REASON,
        )
    reason = _truncate_text(reason, max_chars=max(8, int(max_reason_chars)))

    return ParseAdviceResult(
        action=action,
        reason=reason,
        parse_ok=True,
        validated_ok=True,
        error_type=None,
    )


def _parse_json_object(text: str) -> Any | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    first = raw.find("{")
    last = raw.rfind("}")
    if 0 <= first < last:
        snippet = raw[first : last + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."
