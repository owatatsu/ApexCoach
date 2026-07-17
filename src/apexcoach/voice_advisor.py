from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from queue import Queue
from threading import Condition, Event, Lock, Thread, current_thread
from typing import Callable, Protocol

from apexcoach.config import VoiceConfig
from apexcoach.display_text import action_label, localize_reason
from apexcoach.models import Action, ArbiterResult, Decision


LOGGER = logging.getLogger(__name__)
_EVENT_STOP = object()
_WORKER_JOIN_TIMEOUT_SECONDS = 3.0
_EVENT_DISPATCHER_JOIN_TIMEOUT_SECONDS = 3.0


class _Speaker(Protocol):
    def speak(self, text: str) -> None: ...

    def close(self) -> None: ...


_DEFAULT_URGENT_ACTIONS = {Action.RETREAT, Action.TAKE_COVER}
_ACTION_PRIORITIES = {
    Action.RETREAT: 1,
    Action.TAKE_COVER: 2,
    Action.HEAL: 3,
    Action.TAKE_HIGH_GROUND: 4,
    Action.PUSH: 5,
}
_SHORT_LABELS = {
    Action.RETREAT: "退避",
    Action.TAKE_COVER: "遮蔽へ",
    Action.HEAL: "今、回復",
    Action.TAKE_HIGH_GROUND: "高所へ",
    Action.PUSH: "敵ダウン、詰める",
}


@dataclass(slots=True)
class VoiceMessage:
    action: Action
    text: str
    priority: int
    detected_timestamp: float
    enqueue_monotonic: float
    expiry_monotonic: float
    reason: str = ""
    source: str = "rule"
    urgent: bool = False
    sequence: int = 0

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expiry_monotonic


class VoiceAdvisor:
    """Non-blocking, priority-aware speech queue for tactical advice."""

    def __init__(
        self,
        config: VoiceConfig,
        *,
        speaker_factory: Callable[[VoiceConfig], _Speaker] | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.config = config
        self._speaker_factory = speaker_factory or _build_speaker
        self._event_sink = event_sink
        self._queue: list[VoiceMessage] = []
        self._condition = Condition()
        self._stop_event = Event()
        self._startup_event = Event()
        self._health_lock = Lock()
        self._health_state = "disabled" if not config.enabled else "starting"
        self._disabled_reason: str | None = None
        self._closed = False
        self._enabled = bool(config.enabled)
        self._thread: Thread | None = None
        self._event_queue: Queue[object] = Queue()
        self._event_thread: Thread | None = None
        self._event_stop_lock = Lock()
        self._event_dispatcher_stop_requested = False
        self._sequence = 0
        self._last_enqueue_ts: float | None = None
        self._last_text: str | None = None
        self._last_text_ts: float | None = None
        self._last_action_ts: dict[Action, float] = {}

        if self._enabled and self._event_sink is not None:
            self._event_thread = Thread(
                target=self._run_event_dispatcher,
                name="apexcoach-voice-events",
                daemon=True,
            )
            self._event_thread.start()

        if self._enabled:
            # The worker owns the complete TTS lifecycle. The main thread waits
            # only for the startup health result, never for a speaker object.
            self._thread = Thread(
                target=self._run,
                name="apexcoach-voice-advisor",
                daemon=True,
            )
            self._thread.start()
            timeout = max(0.01, float(config.startup_timeout_seconds))
            if not self._startup_event.wait(timeout=timeout):
                self._disable("speaker startup timed out")
                self._stop_event.set()
                self._drop_pending("startup_timeout")
                with self._condition:
                    self._condition.notify_all()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    @property
    def health_state(self) -> str:
        with self._health_lock:
            return self._health_state

    @property
    def queue_depth(self) -> int:
        with self._condition:
            return len(self._queue)

    def maybe_speak(
        self,
        *,
        decision: Decision,
        arbiter: ArbiterResult,
        timestamp: float,
        urgent: bool | None = None,
    ) -> str | None:
        if not self._enabled or not arbiter.emitted or arbiter.action == Action.NONE:
            return None

        action = arbiter.action
        reason = decision.reason if decision.action == action else ""
        message_urgent = (
            bool(decision.meta.get("urgent", False))
            if urgent is None
            else bool(urgent)
        )
        if urgent is None and action in _DEFAULT_URGENT_ACTIONS:
            message_urgent = True
        text = build_voice_text(
            action=action,
            reason=reason,
            include_reason=self.config.include_reason,
            realtime_short_mode=self.config.realtime_short_mode,
        )
        if not text or not self._allow_enqueue(action, text, timestamp, message_urgent):
            return None

        now = time.monotonic()
        message = VoiceMessage(
            action=action,
            text=text,
            priority=action_priority(action),
            detected_timestamp=float(timestamp),
            enqueue_monotonic=now,
            expiry_monotonic=now + self._ttl_for(message_urgent),
            reason=reason,
            source=str(decision.meta.get("source", "rule")),
            urgent=message_urgent,
            sequence=self._sequence,
        )
        self._sequence += 1
        if not self._enqueue(message):
            return None

        self._last_enqueue_ts = timestamp
        self._last_text = text
        self._last_text_ts = timestamp
        self._last_action_ts[action] = timestamp
        return text

    def close(self) -> None:
        with self._health_lock:
            if self._closed:
                thread = self._thread
            else:
                self._closed = True
                thread = self._thread
                if self._health_state not in {"error", "disabled"}:
                    self._health_state = "closing"

        self._stop_event.set()
        self._drop_pending("close")
        with self._condition:
            self._condition.notify_all()

        if thread is not None and thread is not current_thread():
            thread.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
        if thread is not None and thread.is_alive():  # pragma: no cover
            LOGGER.warning("Voice advisor worker did not stop within 3 seconds.")
            return
        self._thread = None
        self._stop_event_dispatcher()

    def _allow_enqueue(
        self,
        action: Action,
        text: str,
        timestamp: float,
        urgent: bool,
    ) -> bool:
        min_interval = max(0.0, float(self.config.min_interval_seconds))
        bypass = urgent and bool(self.config.emergency_interval_bypass)
        if (
            not bypass
            and self._last_enqueue_ts is not None
            and timestamp - self._last_enqueue_ts < min_interval
        ):
            return False

        cooldown = max(0.0, float(self.config.same_text_cooldown_seconds))
        last_action = self._last_action_ts.get(action)
        if last_action is not None and timestamp - last_action < cooldown:
            return False
        return not (
            self._last_text == text
            and self._last_text_ts is not None
            and timestamp - self._last_text_ts < cooldown
        )

    def _ttl_for(self, urgent: bool) -> float:
        configured = (
            self.config.emergency_message_ttl_seconds
            if urgent
            else self.config.normal_message_ttl_seconds
        )
        return max(0.001, float(configured))

    def _enqueue(self, message: VoiceMessage) -> bool:
        with self._condition:
            if self._stop_event.is_set() or self._closed:
                self._emit_event(message, "dropped", dropped_reason="closed")
                return False

            if len(self._queue) >= max(1, int(self.config.max_queue_size)):
                victim = max(
                    self._queue,
                    key=lambda item: (
                        item.priority,
                        -item.enqueue_monotonic,
                        -item.sequence,
                    ),
                )
                if victim.priority < message.priority:
                    self._emit_event(
                        message,
                        "dropped",
                        dropped_reason="queue_full_lower_priority",
                    )
                    return False
                self._queue.remove(victim)
                self._emit_event(victim, "dropped", dropped_reason="queue_full_eviction")

            self._queue.append(message)
            self._queue.sort(key=lambda item: (item.priority, item.sequence))
            self._emit_event(message, "enqueued")
            self._condition.notify()
            return True

    def _run(self) -> None:
        speaker: _Speaker | None = None
        try:
            try:
                speaker = self._speaker_factory(self.config)
            except Exception as exc:  # pragma: no cover - host TTS dependent
                self._disable(f"speaker initialization failed: {exc}")
                self._startup_event.set()
                self._drop_pending("startup_failed")
                return

            if self._stop_event.is_set() or not self._enabled:
                self._startup_event.set()
                self._drop_pending("startup_cancelled")
                return

            with self._health_lock:
                self._health_state = "ready"
            self._startup_event.set()

            while True:
                with self._condition:
                    while not self._queue and not self._stop_event.is_set():
                        self._condition.wait(timeout=0.25)
                    if self._stop_event.is_set():
                        return
                    message = self._queue.pop(0)

                if message.expired:
                    self._emit_event(message, "dropped", dropped_reason="expired")
                    continue

                start_latency_ms = max(
                    0.0, (time.monotonic() - message.enqueue_monotonic) * 1000.0
                )
                self._emit_event(
                    message,
                    "started",
                    latency_ms=round(start_latency_ms, 1),
                )
                try:
                    speaker.speak(message.text)
                except Exception as exc:  # pragma: no cover - runtime safety
                    self._emit_event(
                        message,
                        "failed",
                        latency_ms=round(start_latency_ms, 1),
                        error=str(exc),
                    )
                    self._disable(f"speech failed: {exc}")
                    return
                self._emit_event(
                    message,
                    "completed",
                    latency_ms=round(start_latency_ms, 1),
                )
        finally:
            if speaker is not None:
                try:
                    speaker.close()
                except Exception:  # pragma: no cover - runtime safety
                    LOGGER.debug("Voice speaker close failed.", exc_info=True)
            with self._health_lock:
                if self._health_state not in {"error", "disabled"}:
                    self._health_state = "closed"
                if self._thread is current_thread():
                    self._thread = None
            # This is deliberately in the worker's finally block. If close()
            # timed out while speak() was blocked, the dispatcher is stopped
            # only after the worker has emitted its final lifecycle event.
            self._stop_event_dispatcher()

    def _drop_pending(self, reason: str) -> None:
        with self._condition:
            pending = list(self._queue)
            self._queue.clear()
            for message in pending:
                self._emit_event(message, "dropped", dropped_reason=reason)

    def _emit_event(
        self,
        message: VoiceMessage,
        status: str,
        *,
        dropped_reason: str | None = None,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        event: dict[str, object] = {
            "event": status,
            "action": message.action.value,
            "text": message.text,
            "priority": message.priority,
            "urgent": message.urgent,
            "detected_timestamp": message.detected_timestamp,
            "enqueue_monotonic": message.enqueue_monotonic,
            "expiry_monotonic": message.expiry_monotonic,
            "reason": message.reason,
            "source": message.source,
            "queue_depth": self.queue_depth,
            "latency_ms": round(float(latency_ms), 1),
        }
        if dropped_reason:
            event["dropped_reason"] = dropped_reason
        if error:
            event["error"] = error
        # Queueing is intentionally the only operation on the frame/voice
        # caller path. The external sink runs in _run_event_dispatcher().
        with self._event_stop_lock:
            if self._event_dispatcher_stop_requested:
                return
            self._event_queue.put_nowait(event)

    def _run_event_dispatcher(self) -> None:
        try:
            while True:
                item = self._event_queue.get()
                if item is _EVENT_STOP:
                    return
                if not isinstance(item, dict) or self._event_sink is None:
                    continue
                try:
                    self._event_sink(item)
                except Exception:  # pragma: no cover - logging must not break audio
                    LOGGER.exception("Voice event logging failed.")
        finally:
            with self._event_stop_lock:
                if self._event_thread is current_thread():
                    self._event_thread = None

    def _stop_event_dispatcher(self) -> None:
        with self._event_stop_lock:
            thread = self._event_thread
            if thread is None:
                return
            if not self._event_dispatcher_stop_requested:
                self._event_dispatcher_stop_requested = True
                # Queue order guarantees every event emitted before this
                # sentinel is delivered before the dispatcher exits.
                self._event_queue.put(_EVENT_STOP)
        if thread is not current_thread():
            thread.join(timeout=_EVENT_DISPATCHER_JOIN_TIMEOUT_SECONDS)
        if thread.is_alive():  # pragma: no cover - external sink dependent
            LOGGER.warning("Voice event dispatcher did not stop within 3 seconds.")
        else:
            with self._event_stop_lock:
                if self._event_thread is thread:
                    self._event_thread = None

    def _disable(self, reason: str) -> None:
        self._enabled = False
        self._disabled_reason = reason
        with self._health_lock:
            self._health_state = "error"
        LOGGER.warning("Voice advice disabled: %s", reason)


def action_priority(action: Action) -> int:
    return _ACTION_PRIORITIES.get(action, 0)


def build_voice_text(
    action: Action,
    reason: str,
    include_reason: bool,
    realtime_short_mode: bool = False,
) -> str:
    label = _SHORT_LABELS.get(action, action_label(action)) if realtime_short_mode else action_label(action)
    if not include_reason:
        return label
    localized_reason = localize_reason(reason)
    if not localized_reason:
        return label
    return f"{label}。{localized_reason[:64]}"


class _Pyttsx3Speaker:
    def __init__(self, config: VoiceConfig) -> None:
        try:
            import pyttsx3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyttsx3 is required for voice advice. Install `pip install -e .[voice]`."
            ) from exc

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", max(80, int(config.rate)))
            self._engine.setProperty("volume", max(0.0, min(1.0, float(config.volume))))
            voice_id = (config.voice_id or "").strip()
            if voice_id:
                voices = self._engine.getProperty("voices")
                voice_ids = {
                    str(getattr(voice, "id", "")).strip()
                    for voice in (voices or [])
                }
                if voice_ids and voice_id not in voice_ids:
                    raise RuntimeError(f"Configured voice ID is unavailable: {voice_id}")
                self._engine.setProperty("voice", voice_id)
        except Exception:
            try:
                self._engine.stop()
            except Exception:
                pass
            raise

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    def close(self) -> None:
        self._engine.stop()


def _build_speaker(config: VoiceConfig) -> _Speaker:
    backend = (config.backend or "pyttsx3").strip().lower()
    if backend in {"pyttsx3", "auto"}:
        return _Pyttsx3Speaker(config)
    raise ValueError(f"Unsupported voice backend: {config.backend!r}")
