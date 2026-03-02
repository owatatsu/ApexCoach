from __future__ import annotations

import math

from apexcoach.config import ArbiterConfig
from apexcoach.models import Action, ArbiterResult, Decision


class ActionArbiter:
    def __init__(self, config: ArbiterConfig) -> None:
        self.c = config
        self._current_action = Action.NONE
        self._current_since = 0.0
        self._last_emit: dict[Action, float] = {}
        self._last_retreat_time: float | None = None

    @property
    def current_action(self) -> Action:
        return self._current_action

    @property
    def current_since(self) -> float:
        return self._current_since

    def arbitrate(self, decision: Decision, timestamp: float) -> ArbiterResult:
        candidate = decision.action
        debug_notes: list[str] = []

        if (
            candidate == Action.PUSH
            and self._last_retreat_time is not None
            and timestamp - self._last_retreat_time
            < self.c.push_block_after_retreat_seconds
        ):
            candidate = Action.NONE
            debug_notes.append("push_block_after_retreat")

        if (
            self._current_action == Action.RETREAT
            and candidate != Action.RETREAT
            and self._last_retreat_time is not None
            and timestamp - self._last_retreat_time < self.c.retreat_hold_seconds
        ):
            candidate = Action.RETREAT
            debug_notes.append("retreat_hysteresis_hold")

        changed = candidate != self._current_action
        if changed:
            self._current_action = candidate
            self._current_since = timestamp
            if candidate == Action.RETREAT:
                self._last_retreat_time = timestamp

        emitted = False
        if self._current_action != Action.NONE:
            if changed and self._current_action == Action.RETREAT:
                emitted = True
            else:
                last_emit = self._last_emit.get(self._current_action, -math.inf)
                if timestamp - last_emit >= self.c.same_action_cooldown_seconds:
                    emitted = True

        if emitted:
            self._last_emit[self._current_action] = timestamp

        return ArbiterResult(
            action=self._current_action,
            emitted=emitted,
            reason=decision.reason,
            source_action=decision.action,
            debug_notes=debug_notes,
        )
