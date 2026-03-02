from __future__ import annotations

from apexcoach.config import OverlayConfig
from apexcoach.models import Action

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


class OverlayRenderer:
    def __init__(self, config: OverlayConfig) -> None:
        self.c = config

    def render(self, frame, action: Action, reason: str) -> object:
        if not self.c.enabled or cv2 is None:
            return frame

        out = frame.copy()
        frame_h, frame_w = out.shape[:2]

        if action != Action.NONE:
            color = _action_color(action)
            x, y = _resolve_text_origin(self.c, frame_w=frame_w, frame_h=frame_h)
            panel_width = max(360, int(self.c.panel_width))
            cv2.rectangle(
                out,
                (x - 12, y - 34),
                (x + panel_width, y + 42),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                out,
                f"ACTION: {action.value}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )
            if self.c.show_reason:
                cv2.putText(
                    out,
                    reason,
                    (x, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

        if self.c.show_window:
            cv2.imshow(self.c.window_name, out)
            cv2.waitKey(1)

        return out

    def close(self) -> None:
        if self.c.show_window and cv2 is not None:
            cv2.destroyAllWindows()


def _action_color(action: Action) -> tuple[int, int, int]:
    if action == Action.RETREAT:
        return (60, 80, 240)
    if action == Action.HEAL:
        return (80, 200, 120)
    if action == Action.PUSH:
        return (240, 180, 40)
    return (220, 220, 220)


def _resolve_text_origin(
    config: OverlayConfig, frame_w: int, frame_h: int
) -> tuple[int, int]:
    margin_x = max(12, int(config.margin_x))
    panel_width = max(360, int(config.panel_width))

    if config.position == "right_center":
        x = frame_w - panel_width - margin_x
        y = (frame_h // 2) + int(config.offset_y)
    elif config.position == "left_top":
        x = margin_x
        y = 54 + int(config.offset_y)
    else:
        x = int(config.text_x)
        y = int(config.text_y)

    x = min(max(12, x), max(12, frame_w - panel_width - 12))
    y = min(max(34, y), max(34, frame_h - 42))
    return x, y
