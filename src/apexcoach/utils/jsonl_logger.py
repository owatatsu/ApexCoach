from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._lock = Lock()
        self._handle = None

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            return
        try:
            line = json.dumps(record, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                self._handle.write(line + "\n")
                self._handle.flush()
            except OSError:
                return
