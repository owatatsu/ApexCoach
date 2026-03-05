from apexcoach.models import Action
from apexcoach.providers.response_parser import ParseErrorType, parse_advice_text


def test_parse_advice_json_ok() -> None:
    result = parse_advice_text(
        raw_text='{"action":"RETREAT","reason":"被弾が大きいので引く"}',
        candidate_actions=[Action.RETREAT, Action.NONE],
        max_reason_chars=48,
    )
    assert result.validated_ok is True
    assert result.action == Action.RETREAT


def test_parse_advice_extract_json_from_wrapped_text() -> None:
    raw = 'note: {"action":"NONE","reason":"不確実なので維持"} end'
    result = parse_advice_text(
        raw_text=raw,
        candidate_actions=[Action.NONE, Action.PUSH],
        max_reason_chars=48,
    )
    assert result.validated_ok is True
    assert result.action == Action.NONE


def test_parse_advice_rejects_non_candidate_action() -> None:
    result = parse_advice_text(
        raw_text='{"action":"PUSH","reason":"行くべき"}',
        candidate_actions=[Action.RETREAT, Action.NONE],
        max_reason_chars=48,
    )
    assert result.validated_ok is False
    assert result.error_type == ParseErrorType.ACTION_NOT_CANDIDATE


def test_parse_advice_rejects_missing_reason() -> None:
    result = parse_advice_text(
        raw_text='{"action":"RETREAT"}',
        candidate_actions=[Action.RETREAT, Action.NONE],
        max_reason_chars=48,
    )
    assert result.validated_ok is False
    assert result.error_type == ParseErrorType.MISSING_REASON


def test_parse_advice_truncates_reason() -> None:
    raw = '{"action":"RETREAT","reason":"' + ("a" * 80) + '"}'
    result = parse_advice_text(
        raw_text=raw,
        candidate_actions=[Action.RETREAT, Action.NONE],
        max_reason_chars=20,
    )
    assert result.validated_ok is True
    assert result.reason is not None
    assert len(result.reason) <= 20
