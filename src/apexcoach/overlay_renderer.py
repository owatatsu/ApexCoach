from __future__ import annotations

import sys
from pathlib import Path

from apexcoach.config import OverlayConfig
from apexcoach.display_text import action_label, format_instruction_line, is_llm_label
from apexcoach.models import Action

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    np = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - runtime dependency
    Image = None
    ImageDraw = None
    ImageFont = None


_PIL_FONT_CACHE: dict[int, object] = {}
_PIL_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    r"C:\Windows\Fonts\YuGothR.ttc",
]


class OverlayRenderer:
    def __init__(self, config: OverlayConfig) -> None:
        self.c = config
        self._window_ready = False
        self._hud_hwnd: int | None = None
        self._hud_style_applied = False
        self._active_lines: list[str] = []
        self._hold_until_ts: float = -1.0
        self._active_llm_message: str | None = None
        self._llm_hold_until_ts: float = -1.0

    def render(
        self,
        frame,
        action: Action,
        reason: str,
        *,
        timestamp: float | None = None,
        decision_lines: list[str] | None = None,
        llm_message: str | None = None,
        roi_boxes: dict[str, tuple[int, int, int, int]] | None = None,
    ) -> object:
        if not self.c.enabled or cv2 is None:
            return frame

        out = frame.copy()
        frame_h, frame_w = out.shape[:2]
        lines = self._resolve_lines(action, reason, timestamp, decision_lines)
        llm_line = self._resolve_llm_message(timestamp=timestamp, llm_message=llm_message)
        draw_lines = list(lines)
        if draw_lines and llm_line:
            draw_lines.append(f"補足 | {llm_line}")

        if self.c.debug_show_rois and roi_boxes:
            out = _draw_roi_debug_overlay(out, self.c, roi_boxes)

        if draw_lines:
            out = _draw_lines_on_frame(out, self.c, draw_lines)

        if self.c.show_window:
            if self.c.window_mode == "hud":
                self._show_hud_window(
                    frame_w=frame_w,
                    frame_h=frame_h,
                    lines=draw_lines,
                )
            else:
                self._show_frame_window(out)
            cv2.waitKey(1)

        return out

    def close(self) -> None:
        if self.c.show_window and cv2 is not None:
            cv2.destroyAllWindows()

    def _resolve_lines(
        self,
        action: Action,
        reason: str,
        timestamp: float | None,
        decision_lines: list[str] | None,
    ) -> list[str]:
        lines: list[str] = []
        max_lines = max(1, int(self.c.max_lines))
        hold_seconds = max(0.0, float(self.c.display_hold_seconds))

        if decision_lines:
            lines = [line for line in decision_lines if line][:max_lines]
            self._active_lines = lines
            if timestamp is not None:
                self._hold_until_ts = timestamp + hold_seconds

        if not lines and action != Action.NONE:
            lines = [format_instruction_line(action, reason)]
            self._active_lines = lines
            if timestamp is not None:
                self._hold_until_ts = timestamp + hold_seconds

        if lines:
            return lines

        if timestamp is not None and self._active_lines and timestamp <= self._hold_until_ts:
            return self._active_lines[:max_lines]

        return []

    def _show_frame_window(self, frame) -> None:
        cv2.imshow(self.c.window_name, frame)
        if self.c.window_always_on_top:
            try:
                cv2.setWindowProperty(self.c.window_name, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass

    def _show_hud_window(
        self,
        frame_w: int,
        frame_h: int,
        lines: list[str],
    ) -> None:
        if np is None:
            return

        panel_width = max(360, int(self.c.panel_width))
        pad = 12
        text_scale = max(0.5, float(self.c.text_scale))
        line_step = int(30 * text_scale)
        line_step = max(20, line_step)
        line_count = max(1, len(lines))
        hud_w = panel_width + (pad * 2)
        hud_h = 14 + (line_count * line_step) + 16

        key_bgr = (255, 0, 255)
        hud = np.full((hud_h, hud_w, 3), key_bgr, dtype=np.uint8)
        if lines:
            hud = _draw_lines_on_frame(
                hud,
                self.c,
                lines,
                origin=(pad, 14 + int(18 * text_scale)),
                panel_width=panel_width,
                apply_alpha=False,
            )

        cv2.namedWindow(self.c.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.c.window_name, hud)

        x, y = _resolve_hud_window_origin(
            self.c, frame_w=frame_w, frame_h=frame_h, hud_w=hud_w, hud_h=hud_h
        )
        cv2.resizeWindow(self.c.window_name, hud_w, hud_h)
        cv2.moveWindow(self.c.window_name, x, y)

        if self.c.window_always_on_top:
            try:
                cv2.setWindowProperty(self.c.window_name, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass

        if not self._window_ready:
            self._window_ready = True
        if sys.platform == "win32" and not self._hud_style_applied:
            self._hud_hwnd = _find_window_handle(self.c.window_name)
            if self._hud_hwnd:
                _style_hud_window(
                    hwnd=self._hud_hwnd,
                    transparent=self.c.window_transparent,
                    click_through=self.c.window_click_through,
                    background_alpha=self.c.background_alpha,
                )
                self._hud_style_applied = True

    def _resolve_llm_message(
        self,
        *,
        timestamp: float | None,
        llm_message: str | None,
    ) -> str | None:
        hold_seconds = max(0.0, float(self.c.display_hold_seconds))
        message = (llm_message or "").strip()

        if message:
            self._active_llm_message = message
            if timestamp is not None:
                self._llm_hold_until_ts = timestamp + hold_seconds
            return message

        if (
            timestamp is not None
            and self._active_llm_message
            and timestamp <= self._llm_hold_until_ts
        ):
            return self._active_llm_message

        return None


def _draw_lines_on_frame(
    frame,
    config: OverlayConfig,
    lines: list[str],
    *,
    origin: tuple[int, int] | None = None,
    panel_width: int | None = None,
    apply_alpha: bool = True,
):
    if cv2 is None or np is None:
        return frame

    out = frame.copy()
    frame_h, frame_w = out.shape[:2]
    text_scale = max(0.5, float(config.text_scale))
    line_step = int(30 * text_scale)
    line_step = max(20, line_step)

    if origin is None:
        x, y = _resolve_text_origin(config, frame_w=frame_w, frame_h=frame_h)
    else:
        x, y = origin

    width = max(360, int(panel_width if panel_width is not None else config.panel_width))
    height = 16 + (len(lines) * line_step) + 12
    x1 = max(0, x - 10)
    y1 = max(0, y - int(24 * text_scale))
    x2 = min(frame_w - 1, x + width)
    y2 = min(frame_h - 1, y1 + height)

    alpha = max(0.0, min(1.0, float(config.background_alpha)))
    if apply_alpha:
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    else:
        cv2.rectangle(out, (x1, y1), (x2, y2), (18, 18, 18), -1)

    if _needs_unicode_render(lines) and _can_render_with_pillow():
        out = _draw_lines_with_pillow(
            out=out,
            lines=lines,
            x=x,
            y=y,
            line_step=line_step,
            text_scale=text_scale,
        )
    else:
        for idx, line in enumerate(lines):
            ty = y + (idx * line_step)
            color = _line_color(line)
            cv2.putText(
                out,
                line,
                (x, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95 * text_scale,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def _draw_lines_with_pillow(
    *,
    out,
    lines: list[str],
    x: int,
    y: int,
    line_step: int,
    text_scale: float,
):
    if cv2 is None or np is None or Image is None or ImageDraw is None:
        return out

    font_px = max(16, int(round(25 * text_scale)))
    font = _load_pillow_font(font_px)
    if font is None:
        return out

    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    text_y_offset = int(font_px * 0.88)
    for idx, line in enumerate(lines):
        ty = y + (idx * line_step)
        py = ty - text_y_offset
        bgr = _line_color(line)
        rgb_color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        draw.text(
            (x, py),
            line,
            font=font,
            fill=rgb_color,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_roi_debug_overlay(
    frame,
    config: OverlayConfig,
    roi_boxes: dict[str, tuple[int, int, int, int]],
):
    if cv2 is None:
        return frame

    out = frame.copy()
    overlay = out.copy()
    alpha = max(0.05, min(0.95, float(config.debug_roi_alpha)))
    palette = [
        (255, 120, 80),   # BGR
        (90, 210, 250),
        (100, 220, 120),
        (240, 180, 80),
        (220, 110, 220),
        (130, 220, 220),
    ]

    for idx, name in enumerate(sorted(roi_boxes.keys())):
        x1, y1, x2, y2 = roi_boxes[name]
        color = palette[idx % len(palette)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if config.debug_show_roi_labels:
            label_y = max(14, y1 - 6)
            cv2.putText(
                out,
                name,
                (x1 + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    return out


def _needs_unicode_render(lines: list[str]) -> bool:
    for line in lines:
        if any(ord(ch) > 127 for ch in line):
            return True
    return False


def _can_render_with_pillow() -> bool:
    return Image is not None and ImageDraw is not None and ImageFont is not None


def _load_pillow_font(size_px: int):
    if ImageFont is None:
        return None

    key = int(size_px)
    cached = _PIL_FONT_CACHE.get(key)
    if cached is not None:
        return cached

    for path in _PIL_FONT_CANDIDATES:
        try:
            if Path(path).exists():
                font = ImageFont.truetype(path, key)
                _PIL_FONT_CACHE[key] = font
                return font
        except Exception:
            continue
    return None


def _line_color(line: str) -> tuple[int, int, int]:
    action_token = line.split("|", 1)[0].strip()
    if action_token in {Action.RETREAT.value, action_label(Action.RETREAT)}:
        return (60, 80, 240)
    if action_token in {Action.TAKE_COVER.value, action_label(Action.TAKE_COVER)}:
        return (90, 210, 245)
    if action_token in {
        Action.TAKE_HIGH_GROUND.value,
        action_label(Action.TAKE_HIGH_GROUND),
    }:
        return (230, 200, 90)
    if action_token in {Action.HEAL.value, action_label(Action.HEAL)}:
        return (80, 200, 120)
    if action_token in {Action.PUSH.value, action_label(Action.PUSH)}:
        return (240, 180, 40)
    if is_llm_label(action_token):
        return (220, 220, 255)
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


def _resolve_hud_window_origin(
    config: OverlayConfig, frame_w: int, frame_h: int, hud_w: int, hud_h: int
) -> tuple[int, int]:
    margin_x = max(12, int(config.margin_x))
    if config.position == "right_center":
        x = frame_w - hud_w - margin_x
        y = (frame_h // 2) - (hud_h // 2) + int(config.offset_y)
    elif config.position == "left_top":
        x = margin_x
        y = 32 + int(config.offset_y)
    else:
        x = int(config.text_x)
        y = int(config.text_y)
    return max(0, x), max(0, y)


def _find_window_handle(window_name: str) -> int | None:
    try:
        import ctypes

        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        return int(hwnd) if hwnd else None
    except Exception:
        return None


def _style_hud_window(
    hwnd: int,
    transparent: bool,
    click_through: bool,
    background_alpha: float,
) -> None:
    try:
        import ctypes

        user32 = ctypes.windll.user32

        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        WS_CAPTION = 0x00C00000
        WS_THICKFRAME = 0x00040000
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        LWA_COLORKEY = 0x00000001
        LWA_ALPHA = 0x00000002
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010
        SWP_FRAMECHANGED = 0x0020

        style = int(user32.GetWindowLongW(hwnd, GWL_STYLE))
        style &= ~WS_CAPTION
        style &= ~WS_THICKFRAME
        user32.SetWindowLongW(hwnd, GWL_STYLE, style)

        ex_style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
        ex_style |= WS_EX_TOOLWINDOW
        if transparent:
            ex_style |= WS_EX_LAYERED
        if click_through:
            ex_style |= WS_EX_TRANSPARENT
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)

        if transparent:
            # Use magenta key for outside region + global alpha for panel/transparency feel.
            # COLORREF for magenta (R=255,G=0,B=255).
            colorkey_magenta = 0x00FF00FF
            alpha = int(max(0.0, min(1.0, background_alpha)) * 255)
            user32.SetLayeredWindowAttributes(
                hwnd,
                colorkey_magenta,
                alpha,
                LWA_COLORKEY | LWA_ALPHA,
            )

        user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
    except Exception:
        return
