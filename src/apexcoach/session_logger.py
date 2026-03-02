from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from apexcoach.models import Action, ArbiterResult, Decision, FrameEvents, FramePacket, GameState


class SessionLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._handle = None

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def log_frame(
        self,
        packet: FramePacket,
        state: GameState,
        events: FrameEvents,
        decision: Decision,
        arbiter: ArbiterResult,
        llm_reason: str | None = None,
    ) -> None:
        if self._handle is None:
            return

        record = {
            "timestamp": packet.timestamp,
            "frame_index": packet.frame_index,
            "state": _serialize(state),
            "events": _serialize(events),
            "decision": _serialize(decision),
            "arbiter": _serialize(arbiter),
            "llm_reason": llm_reason,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _serialize(value: Any) -> Any:
    if isinstance(value, Action):
        return value.value
    if is_dataclass(value):
        return {k: _serialize(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value
