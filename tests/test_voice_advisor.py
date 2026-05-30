from threading import Event

from apexcoach.config import VoiceConfig
from apexcoach.display_text import action_label
from apexcoach.models import Action, ArbiterResult, Decision
from apexcoach.voice_advisor import VoiceAdvisor, build_voice_text


class _FakeSpeaker:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.spoken = Event()
        self.closed = False

    def speak(self, text: str) -> None:
        self.messages.append(text)
        self.spoken.set()

    def close(self) -> None:
        self.closed = True


def _emitted(action: Action) -> ArbiterResult:
    return ArbiterResult(
        action=action,
        emitted=True,
        reason="rule",
        source_action=action,
    )


def test_build_voice_text_can_keep_action_short() -> None:
    text = build_voice_text(
        action=Action.RETREAT,
        reason="High incoming damage in last 1s.",
        include_reason=False,
    )

    assert text == action_label(Action.RETREAT)


def test_voice_advisor_speaks_emitted_advice_without_blocking() -> None:
    speaker = _FakeSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(enabled=True, min_interval_seconds=0.0),
        speaker_factory=lambda config: speaker,
    )

    spoken = advisor.maybe_speak(
        decision=Decision(Action.HEAL, "Safe window and low HP+Shield."),
        arbiter=_emitted(Action.HEAL),
        timestamp=1.0,
    )

    assert spoken
    assert speaker.spoken.wait(timeout=1.0)
    advisor.close()
    assert speaker.messages == [spoken]
    assert speaker.closed is True


def test_voice_advisor_suppresses_unemitted_and_repeated_text() -> None:
    speaker = _FakeSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(
            enabled=True,
            min_interval_seconds=0.0,
            same_text_cooldown_seconds=5.0,
        ),
        speaker_factory=lambda config: speaker,
    )
    decision = Decision(Action.PUSH, "Enemy knock advantage with healthy team.")

    unemitted = advisor.maybe_speak(
        decision=decision,
        arbiter=ArbiterResult(
            action=Action.PUSH,
            emitted=False,
            reason="cooldown",
            source_action=Action.PUSH,
        ),
        timestamp=1.0,
    )
    first = advisor.maybe_speak(
        decision=decision,
        arbiter=_emitted(Action.PUSH),
        timestamp=2.0,
    )
    repeated = advisor.maybe_speak(
        decision=decision,
        arbiter=_emitted(Action.PUSH),
        timestamp=3.0,
    )

    assert unemitted is None
    assert first
    assert repeated is None
    assert speaker.spoken.wait(timeout=1.0)
    advisor.close()
    assert speaker.messages == [first]


def test_voice_advisor_omits_reason_when_arbiter_holds_other_action() -> None:
    speaker = _FakeSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(enabled=True, min_interval_seconds=0.0),
        speaker_factory=lambda config: speaker,
    )

    spoken = advisor.maybe_speak(
        decision=Decision(Action.NONE, "No strong signal."),
        arbiter=_emitted(Action.RETREAT),
        timestamp=1.0,
    )

    assert spoken == action_label(Action.RETREAT)
    assert speaker.spoken.wait(timeout=1.0)
    advisor.close()
