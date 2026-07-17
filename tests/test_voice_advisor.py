import time
from threading import Event, Thread, get_ident

import apexcoach.voice_advisor as voice_advisor_module
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


class _BlockingSpeaker(_FakeSpeaker):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def speak(self, text: str) -> None:
        self.messages.append(text)
        self.started.set()
        self.release.wait(timeout=1.0)


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


def test_emergency_bypasses_normal_interval_and_queues_first() -> None:
    speaker = _BlockingSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(
            enabled=True,
            include_reason=False,
            min_interval_seconds=10.0,
            same_text_cooldown_seconds=0.0,
        ),
        speaker_factory=lambda config: speaker,
    )

    assert advisor.maybe_speak(
        decision=Decision(Action.PUSH, "push"),
        arbiter=_emitted(Action.PUSH),
        timestamp=1.0,
    )
    assert speaker.started.wait(timeout=1.0)
    assert advisor.maybe_speak(
        decision=Decision(Action.RETREAT, "danger"),
        arbiter=_emitted(Action.RETREAT),
        timestamp=1.1,
    )
    speaker.release.set()
    time.sleep(0.05)
    advisor.close()
    assert speaker.messages == ["敵ダウン、詰める", "退避"]


def test_priority_overtakes_pending_lower_priority_message() -> None:
    speaker = _BlockingSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(
            enabled=True,
            include_reason=False,
            min_interval_seconds=0.0,
            same_text_cooldown_seconds=0.0,
            max_queue_size=3,
        ),
        speaker_factory=lambda config: speaker,
    )
    advisor.maybe_speak(
        decision=Decision(Action.PUSH, "push"),
        arbiter=_emitted(Action.PUSH),
        timestamp=0.0,
    )
    assert speaker.started.wait(timeout=1.0)
    advisor.maybe_speak(
        decision=Decision(Action.TAKE_HIGH_GROUND, "height"),
        arbiter=_emitted(Action.TAKE_HIGH_GROUND),
        timestamp=0.1,
    )
    advisor.maybe_speak(
        decision=Decision(Action.RETREAT, "danger"),
        arbiter=_emitted(Action.RETREAT),
        timestamp=0.2,
    )
    speaker.release.set()
    time.sleep(0.05)
    advisor.close()
    assert speaker.messages == ["敵ダウン、詰める", "退避", "高所へ"]


def test_expired_pending_message_is_dropped() -> None:
    speaker = _BlockingSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(
            enabled=True,
            include_reason=False,
            min_interval_seconds=0.0,
            same_text_cooldown_seconds=0.0,
            normal_message_ttl_seconds=0.01,
            emergency_message_ttl_seconds=0.01,
            max_queue_size=3,
        ),
        speaker_factory=lambda config: speaker,
    )
    advisor.maybe_speak(
        decision=Decision(Action.PUSH, "push"),
        arbiter=_emitted(Action.PUSH),
        timestamp=0.0,
    )
    assert speaker.started.wait(timeout=1.0)
    advisor.maybe_speak(
        decision=Decision(Action.TAKE_HIGH_GROUND, "height"),
        arbiter=_emitted(Action.TAKE_HIGH_GROUND),
        timestamp=0.1,
    )
    time.sleep(0.03)
    speaker.release.set()
    time.sleep(0.05)
    advisor.close()
    assert speaker.messages == ["敵ダウン、詰める"]


def test_queue_full_drops_lower_priority_new_message_and_logs_event() -> None:
    speaker = _BlockingSpeaker()
    events: list[dict[str, object]] = []
    advisor = VoiceAdvisor(
        VoiceConfig(
            enabled=True,
            include_reason=False,
            min_interval_seconds=0.0,
            same_text_cooldown_seconds=0.0,
            max_queue_size=2,
        ),
        speaker_factory=lambda config: speaker,
        event_sink=events.append,
    )
    advisor.maybe_speak(
        decision=Decision(Action.PUSH, "push"),
        arbiter=_emitted(Action.PUSH),
        timestamp=0.0,
    )
    assert speaker.started.wait(timeout=1.0)
    advisor.maybe_speak(
        decision=Decision(Action.HEAL, "heal"),
        arbiter=_emitted(Action.HEAL),
        timestamp=0.1,
    )
    advisor.maybe_speak(
        decision=Decision(Action.TAKE_COVER, "cover"),
        arbiter=_emitted(Action.TAKE_COVER),
        timestamp=0.2,
    )
    dropped = advisor.maybe_speak(
        decision=Decision(Action.PUSH, "push again"),
        arbiter=_emitted(Action.PUSH),
        timestamp=0.3,
    )
    speaker.release.set()
    advisor.close()
    assert dropped is None
    assert any(
        e.get("event") == "dropped"
        and e.get("dropped_reason") == "queue_full_lower_priority"
        for e in events
    )


def test_speaker_initialization_failure_disables_voice() -> None:
    advisor = VoiceAdvisor(
        VoiceConfig(enabled=True),
        speaker_factory=lambda config: (_ for _ in ()).throw(RuntimeError("no TTS")),
    )

    assert advisor.enabled is False
    assert "no TTS" in (advisor.disabled_reason or "")
    assert advisor.health_state == "error"
    assert advisor.maybe_speak(
        decision=Decision(Action.RETREAT, "danger"),
        arbiter=_emitted(Action.RETREAT),
        timestamp=1.0,
    ) is None
    advisor.close()


def test_voice_events_record_lifecycle_and_close_is_safe() -> None:
    speaker = _FakeSpeaker()
    events: list[dict[str, object]] = []
    advisor = VoiceAdvisor(
        VoiceConfig(enabled=True, include_reason=False, min_interval_seconds=0.0),
        speaker_factory=lambda config: speaker,
        event_sink=events.append,
    )
    assert advisor.health_state == "ready"
    advisor.maybe_speak(
        decision=Decision(Action.RETREAT, "danger"),
        arbiter=_emitted(Action.RETREAT),
        timestamp=1.0,
    )
    assert speaker.spoken.wait(timeout=1.0)
    advisor.close()
    advisor.close()
    assert [e["event"] for e in events] == [
        "enqueued",
        "started",
        "completed",
    ]


def test_speaker_lifecycle_stays_on_one_worker_thread() -> None:
    factory_thread_ids: list[int] = []
    operation_thread_ids: list[int] = []
    spoken = Event()

    class ThreadRecordingSpeaker:
        def speak(self, text: str) -> None:
            operation_thread_ids.append(get_ident())
            spoken.set()

        def close(self) -> None:
            operation_thread_ids.append(get_ident())

    def factory(config: VoiceConfig) -> ThreadRecordingSpeaker:
        factory_thread_ids.append(get_ident())
        return ThreadRecordingSpeaker()

    advisor = VoiceAdvisor(
        VoiceConfig(enabled=True, include_reason=False, min_interval_seconds=0.0),
        speaker_factory=factory,
    )
    advisor.maybe_speak(
        decision=Decision(Action.RETREAT, "danger"),
        arbiter=_emitted(Action.RETREAT),
        timestamp=1.0,
    )
    assert spoken.wait(timeout=1.0)
    advisor.close()

    assert len(factory_thread_ids) == 1
    assert operation_thread_ids
    assert set(factory_thread_ids + operation_thread_ids) == {factory_thread_ids[0]}


def test_slow_event_sink_does_not_block_maybe_speak() -> None:
    sink_started = Event()
    release_sink = Event()
    events: list[dict[str, object]] = []

    def slow_sink(event: dict[str, object]) -> None:
        sink_started.set()
        release_sink.wait(timeout=1.0)
        events.append(event)

    speaker = _FakeSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(enabled=True, include_reason=False, min_interval_seconds=0.0),
        speaker_factory=lambda config: speaker,
        event_sink=slow_sink,
    )
    started = time.perf_counter()
    advisor.maybe_speak(
        decision=Decision(Action.RETREAT, "danger"),
        arbiter=_emitted(Action.RETREAT),
        timestamp=1.0,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1
    assert sink_started.wait(timeout=1.0)
    assert speaker.spoken.wait(timeout=1.0)
    release_sink.set()
    advisor.close()
    assert [event["event"] for event in events] == [
        "enqueued",
        "started",
        "completed",
    ]


def test_close_drains_voice_events_for_pending_messages() -> None:
    speaker = _BlockingSpeaker()
    events: list[dict[str, object]] = []
    advisor = VoiceAdvisor(
        VoiceConfig(
            enabled=True,
            include_reason=False,
            min_interval_seconds=0.0,
            max_queue_size=3,
        ),
        speaker_factory=lambda config: speaker,
        event_sink=events.append,
    )
    advisor.maybe_speak(
        decision=Decision(Action.PUSH, "push"),
        arbiter=_emitted(Action.PUSH),
        timestamp=1.0,
    )
    assert speaker.started.wait(timeout=1.0)
    advisor.maybe_speak(
        decision=Decision(Action.TAKE_HIGH_GROUND, "height"),
        arbiter=_emitted(Action.TAKE_HIGH_GROUND),
        timestamp=1.1,
    )

    closing = Thread(target=advisor.close)
    closing.start()
    time.sleep(0.05)
    speaker.release.set()
    closing.join(timeout=2.0)

    assert not closing.is_alive()
    assert [event["event"] for event in events].count("enqueued") == 2
    assert any(
        event["event"] == "dropped" and event.get("dropped_reason") == "close"
        for event in events
    )


def test_normal_heal_does_not_bypass_interval_but_critical_heal_does() -> None:
    speaker = _FakeSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(
            enabled=True,
            include_reason=False,
            min_interval_seconds=10.0,
            same_text_cooldown_seconds=5.0,
        ),
        speaker_factory=lambda config: speaker,
    )

    assert advisor.maybe_speak(
        decision=Decision(Action.PUSH, "push"),
        arbiter=_emitted(Action.PUSH),
        timestamp=1.0,
    )
    assert advisor.maybe_speak(
        decision=Decision(Action.HEAL, "normal heal"),
        arbiter=_emitted(Action.HEAL),
        timestamp=1.1,
    ) is None
    assert advisor.maybe_speak(
        decision=Decision(Action.HEAL, "critical heal"),
        arbiter=_emitted(Action.HEAL),
        timestamp=1.2,
        urgent=True,
    )
    assert advisor.maybe_speak(
        decision=Decision(Action.HEAL, "critical heal again"),
        arbiter=_emitted(Action.HEAL),
        timestamp=1.3,
        urgent=True,
    ) is None
    assert advisor._ttl_for(False) == 2.5
    assert advisor._ttl_for(True) == 1.5
    advisor.close()


def test_startup_timeout_disables_voice_without_main_thread_speaker_close() -> None:
    release_factory = Event()
    factory_thread_id: list[int] = []
    close_thread_id: list[int] = []

    class DelayedSpeaker:
        def speak(self, text: str) -> None:
            return None

        def close(self) -> None:
            close_thread_id.append(get_ident())

    def delayed_factory(config: VoiceConfig) -> DelayedSpeaker:
        factory_thread_id.append(get_ident())
        release_factory.wait(timeout=1.0)
        return DelayedSpeaker()

    advisor = VoiceAdvisor(
        VoiceConfig(enabled=True, startup_timeout_seconds=0.02),
        speaker_factory=delayed_factory,
    )
    assert advisor.enabled is False
    assert "timed out" in (advisor.disabled_reason or "")
    release_factory.set()
    advisor.close()
    assert factory_thread_id
    assert close_thread_id == factory_thread_id


def test_close_timeout_event_dispatcher_stops_after_worker_releases(monkeypatch) -> None:
    monkeypatch.setattr(voice_advisor_module, "_WORKER_JOIN_TIMEOUT_SECONDS", 0.02)
    started = Event()
    release = Event()
    events: list[dict[str, object]] = []
    close_thread_ids: list[int] = []

    class NeverEndingSpeaker:
        def speak(self, text: str) -> None:
            started.set()
            release.wait()

        def close(self) -> None:
            close_thread_ids.append(get_ident())

    speaker = NeverEndingSpeaker()
    advisor = VoiceAdvisor(
        VoiceConfig(enabled=True, include_reason=False, min_interval_seconds=0.0),
        speaker_factory=lambda config: speaker,
        event_sink=events.append,
    )
    advisor.maybe_speak(
        decision=Decision(Action.RETREAT, "danger"),
        arbiter=_emitted(Action.RETREAT),
        timestamp=1.0,
    )
    assert started.wait(timeout=1.0)

    advisor.close()
    assert advisor._thread is not None
    assert advisor._event_thread is not None

    release.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and advisor._event_thread is not None:
        time.sleep(0.005)

    assert advisor._thread is None
    assert advisor._event_thread is None
    assert [event["event"] for event in events] == [
        "enqueued",
        "started",
        "completed",
    ]
    assert len(close_thread_ids) == 1
