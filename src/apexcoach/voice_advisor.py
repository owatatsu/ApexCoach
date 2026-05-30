from __future__ import annotations

import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable, Protocol

from apexcoach.config import VoiceConfig
from apexcoach.display_text import action_label, localize_reason
from apexcoach.models import Action, ArbiterResult, Decision


LOGGER = logging.getLogger(__name__)
_STOP = object()


class _Speaker(Protocol):
    def speak(self, text: str) -> None: ...

    def close(self) -> None: ...


class VoiceAdvisor:
    """Non-blocking speech queue for emitted tactical advice."""

    def __init__(
        self,
        config: VoiceConfig,
        *,
        speaker_factory: Callable[[VoiceConfig], _Speaker] | None = None,
    ) -> None:
        self.config = config
        self._speaker_factory = speaker_factory or _build_speaker
        self._queue: Queue[object] = Queue(maxsize=max(1, int(config.max_queue_size)))
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_enqueue_ts: float | None = None
        self._last_text: str | None = None
        self._last_text_ts: float | None = None

        if config.enabled:
            self._thread = Thread(
                target=self._run,
                name="apexcoach-voice-advisor",
                daemon=True,
            )
            self._thread.start()

    def maybe_speak(
        self,
        *,
        decision: Decision,
        arbiter: ArbiterResult,
        timestamp: float,
    ) -> str | None:
        if not self.config.enabled or not arbiter.emitted:
            return None
        if arbiter.action == Action.NONE:
            return None

        text = build_voice_text(
            action=arbiter.action,
            reason=decision.reason if decision.action == arbiter.action else "",
            include_reason=self.config.include_reason,
        )
        if not text or not self._allow_enqueue(text, timestamp):
            return None

        self._offer(text)
        self._last_enqueue_ts = timestamp
        self._last_text = text
        self._last_text_ts = timestamp
        return text

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._offer(_STOP)
        self._thread.join(timeout=3.0)
        self._thread = None

    def _allow_enqueue(self, text: str, timestamp: float) -> bool:
        min_interval = max(0.0, float(self.config.min_interval_seconds))
        if (
            self._last_enqueue_ts is not None
            and timestamp - self._last_enqueue_ts < min_interval
        ):
            return False

        same_text_cooldown = max(0.0, float(self.config.same_text_cooldown_seconds))
        return not (
            self._last_text == text
            and self._last_text_ts is not None
            and timestamp - self._last_text_ts < same_text_cooldown
        )

    def _offer(self, item: object) -> None:
        try:
            self._queue.put_nowait(item)
            return
        except Full:
            pass

        try:
            self._queue.get_nowait()
        except Empty:
            pass

        try:
            self._queue.put_nowait(item)
        except Full:
            return

    def _run(self) -> None:
        speaker: _Speaker | None = None
        try:
            speaker = self._speaker_factory(self.config)
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                if not isinstance(item, str) or not item.strip():
                    continue
                try:
                    speaker.speak(item)
                except Exception as exc:  # pragma: no cover - runtime safety
                    LOGGER.warning("Voice advice failed and was disabled: %s", exc)
                    break
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            LOGGER.warning("Voice advice could not start and was disabled: %s", exc)
        finally:
            if speaker is not None:
                try:
                    speaker.close()
                except Exception:
                    pass


def build_voice_text(action: Action, reason: str, include_reason: bool) -> str:
    label = action_label(action)
    if not include_reason:
        return label
    localized_reason = localize_reason(reason)
    if not localized_reason:
        return label
    return f"{label}。{localized_reason}"


class _Pyttsx3Speaker:
    def __init__(self, config: VoiceConfig) -> None:
        try:
            import pyttsx3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyttsx3 is required for voice advice. Install `pip install -e .[voice]`."
            ) from exc

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", max(80, int(config.rate)))
        self._engine.setProperty("volume", max(0.0, min(1.0, float(config.volume))))
        voice_id = (config.voice_id or "").strip()
        if voice_id:
            self._engine.setProperty("voice", voice_id)

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
