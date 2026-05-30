from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ReportEvent:
    timestamp: float
    action: str
    reason: str
    hp_pct: float
    shield_pct: float
    under_fire: bool
    allies_alive: int
    allies_down: int


@dataclass(slots=True)
class SessionReportData:
    source_log: str
    frames: int = 0
    first_ts: float | None = None
    last_ts: float | None = None
    action_counts: Counter[str] = field(default_factory=Counter)
    emitted_counts: Counter[str] = field(default_factory=Counter)
    under_fire_frames: int = 0
    low_resource_frames: int = 0
    hp_samples: list[tuple[float, float]] = field(default_factory=list)
    shield_samples: list[tuple[float, float]] = field(default_factory=list)
    confidence_samples: list[float] = field(default_factory=list)
    events: list[ReportEvent] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return max(0.0, self.last_ts - self.first_ts)


def build_session_report(log_path: str | Path) -> SessionReportData:
    path = Path(log_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Session log does not exist: {path}")

    data = SessionReportData(source_log=str(path))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = _parse_record(line)
            if record is None:
                continue
            _add_record(data, record)

    if data.frames <= 0:
        raise ValueError(f"Session log has no readable frame records: {path}")
    return data


def write_session_report(
    *,
    log_path: str | Path,
    output_path: str | Path,
) -> str:
    data = build_session_report(log_path)
    rendered = render_session_report(data)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return str(output)


def default_report_output_path(log_path: str | Path) -> Path:
    path = Path(log_path).expanduser()
    return path.with_suffix(".html")


def render_session_report(data: SessionReportData) -> str:
    title = "ApexCoach Session Report"
    action_rows = "\n".join(
        _render_action_row(action, data.action_counts.get(action, 0), data.emitted_counts.get(action, 0))
        for action in _ordered_actions(data.action_counts)
    )
    event_rows = "\n".join(_render_event_row(event) for event in data.events[:80])
    if not event_rows:
        event_rows = '<tr><td colspan="7">No emitted coaching events.</td></tr>'

    chart = _build_vitals_chart(data)
    under_fire_ratio = _ratio(data.under_fire_frames, data.frames)
    low_resource_ratio = _ratio(data.low_resource_frames, data.frames)
    vitals_quality = _build_vitals_quality(data)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #18202a;
  --muted: #657083;
  --line: #d9dee7;
  --hp: #d94c4c;
  --shield: #3f72d8;
  --accent: #20a37f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", system-ui, sans-serif;
  line-height: 1.5;
}}
main {{
  max-width: 1080px;
  margin: 0 auto;
  padding: 28px;
}}
h1 {{ margin: 0 0 4px; font-size: 28px; }}
h2 {{ margin: 0 0 12px; font-size: 18px; }}
.muted {{ color: var(--muted); }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 22px 0;
}}
.metric, section {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(18, 28, 45, 0.04);
}}
.metric {{ padding: 14px 16px; }}
.metric .value {{ display: block; font-size: 26px; font-weight: 700; }}
.metric .label {{ color: var(--muted); font-size: 13px; }}
section {{ padding: 18px; margin: 16px 0; }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 9px 8px;
  text-align: left;
  vertical-align: top;
}}
th {{ color: var(--muted); font-weight: 600; }}
.chart {{
  width: 100%;
  height: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfcfe;
}}
.hp-line {{ fill: none; stroke: var(--hp); stroke-width: 2.5; }}
.shield-line {{ fill: none; stroke: var(--shield); stroke-width: 2.5; }}
.axis {{ stroke: #9aa7b8; stroke-width: 1.2; }}
.grid-line {{ stroke: #e4e8ef; stroke-width: 1; }}
.axis-label {{ fill: var(--muted); font-size: 12px; }}
.axis-title {{ fill: var(--muted); font-size: 13px; font-weight: 600; }}
.legend {{ display: flex; gap: 18px; margin-top: 8px; font-size: 13px; color: var(--muted); }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
.dot.hp {{ background: var(--hp); }}
.dot.shield {{ background: var(--shield); }}
.quality {{
  margin: 0 0 12px;
  padding: 10px 12px;
  border-radius: 6px;
  background: #fff8e6;
  border: 1px solid #f0d38a;
  color: #664b00;
  font-size: 13px;
}}
.quality.ok {{
  background: #edf9f4;
  border-color: #9bd8c1;
  color: #185a46;
}}
</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <div class="muted">{html.escape(data.source_log)}</div>

  <div class="grid">
    {_metric("Frames", str(data.frames))}
    {_metric("Duration", f"{data.duration_sec:.1f}s")}
    {_metric("Under Fire", f"{under_fire_ratio:.1%}")}
    {_metric("Low Resources", f"{low_resource_ratio:.1%}")}
  </div>

  <section>
    <h2>Vitals Timeline</h2>
    {vitals_quality}
    <svg class="chart" viewBox="0 0 920 220" role="img" aria-label="HP and shield timeline">
      {chart}
    </svg>
    <div class="legend"><span><span class="dot hp"></span>HP</span><span><span class="dot shield"></span>Shield</span></div>
  </section>

  <section>
    <h2>Action Counts</h2>
    <table>
      <thead><tr><th>Action</th><th>Frames</th><th>Emitted</th></tr></thead>
      <tbody>{action_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Coaching Events</h2>
    <table>
      <thead><tr><th>Time</th><th>Action</th><th>Reason</th><th>HP</th><th>Shield</th><th>Under Fire</th><th>Team</th></tr></thead>
      <tbody>{event_rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def _add_record(data: SessionReportData, record: dict[str, Any]) -> None:
    ts = float(record.get("timestamp", 0.0) or 0.0)
    if data.first_ts is None:
        data.first_ts = ts
    data.last_ts = ts
    data.frames += 1

    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
    arbiter = record.get("arbiter") if isinstance(record.get("arbiter"), dict) else {}

    hp = _as_float(state.get("hp_pct"), 1.0)
    shield = _as_float(state.get("shield_pct"), 1.0)
    confidence = _optional_float(state.get("vitals_confidence"))
    under_fire = bool(state.get("under_fire", False))
    allies_alive = int(state.get("allies_alive", 3) or 0)
    allies_down = int(state.get("allies_down", 0) or 0)
    action = str(arbiter.get("action") or decision.get("action") or "NONE")
    emitted = bool(arbiter.get("emitted", False))

    data.action_counts[action] += 1
    if emitted:
        data.emitted_counts[action] += 1
    if under_fire:
        data.under_fire_frames += 1
    if hp + shield <= 0.65:
        data.low_resource_frames += 1

    data.hp_samples.append((ts, hp))
    data.shield_samples.append((ts, shield))
    if confidence is not None:
        data.confidence_samples.append(confidence)

    if emitted and action != "NONE":
        data.events.append(
            ReportEvent(
                timestamp=ts,
                action=action,
                reason=str(decision.get("reason") or arbiter.get("reason") or ""),
                hp_pct=hp,
                shield_pct=shield,
                under_fire=under_fire,
                allies_alive=allies_alive,
                allies_down=allies_down,
            )
        )


def _parse_record(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _render_action_row(action: str, frames: int, emitted: int) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(action)}</td>"
        f"<td>{frames}</td>"
        f"<td>{emitted}</td>"
        "</tr>"
    )


def _render_event_row(event: ReportEvent) -> str:
    return (
        "<tr>"
        f"<td>{event.timestamp:.2f}s</td>"
        f"<td>{html.escape(event.action)}</td>"
        f"<td>{html.escape(event.reason)}</td>"
        f"<td>{event.hp_pct:.2f}</td>"
        f"<td>{event.shield_pct:.2f}</td>"
        f"<td>{'yes' if event.under_fire else 'no'}</td>"
        f"<td>{event.allies_alive} up / {event.allies_down} down</td>"
        "</tr>"
    )


def _metric(label: str, value: str) -> str:
    return (
        '<div class="metric">'
        f'<span class="value">{html.escape(value)}</span>'
        f'<span class="label">{html.escape(label)}</span>'
        "</div>"
    )


def _build_vitals_chart(data: SessionReportData) -> str:
    height = 220
    plot = _PlotArea(left=56.0, top=22.0, right=900.0, bottom=172.0)
    duration = max(0.0, data.duration_sec)
    hp_points = _svg_points(data.hp_samples, plot=plot)
    shield_points = _svg_points(data.shield_samples, plot=plot)

    parts: list[str] = []
    for value in (1.0, 0.75, 0.5, 0.25, 0.0):
        y = plot.value_y(value)
        parts.append(f'<line class="grid-line" x1="{plot.left:.1f}" y1="{y:.1f}" x2="{plot.right:.1f}" y2="{y:.1f}"></line>')
        parts.append(f'<text class="axis-label" x="{plot.left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}</text>')

    for tick in _time_ticks(duration):
        x = plot.time_x(tick, duration)
        parts.append(f'<line class="grid-line" x1="{x:.1f}" y1="{plot.top:.1f}" x2="{x:.1f}" y2="{plot.bottom:.1f}"></line>')
        parts.append(f'<text class="axis-label" x="{x:.1f}" y="{plot.bottom + 22:.1f}" text-anchor="middle">{_format_tick_seconds(tick)}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{plot.left:.1f}" y1="{plot.top:.1f}" x2="{plot.left:.1f}" y2="{plot.bottom:.1f}"></line>',
            f'<line class="axis" x1="{plot.left:.1f}" y1="{plot.bottom:.1f}" x2="{plot.right:.1f}" y2="{plot.bottom:.1f}"></line>',
            f'<text class="axis-title" x="{plot.left - 38:.1f}" y="{(plot.top + plot.bottom) / 2:.1f}" transform="rotate(-90 {plot.left - 38:.1f} {(plot.top + plot.bottom) / 2:.1f})" text-anchor="middle">HP / Shield</text>',
            f'<text class="axis-title" x="{(plot.left + plot.right) / 2:.1f}" y="{height - 14:.1f}" text-anchor="middle">Time</text>',
            f'<polyline class="hp-line" points="{hp_points}"></polyline>',
            f'<polyline class="shield-line" points="{shield_points}"></polyline>',
        ]
    )
    return "\n      ".join(parts)


def _build_vitals_quality(data: SessionReportData) -> str:
    hp_zero_ratio = _sample_ratio(data.hp_samples, lambda value: value <= 0.02)
    shield_zero_ratio = _sample_ratio(data.shield_samples, lambda value: value <= 0.02)
    notes: list[str] = []

    if not data.confidence_samples:
        notes.append(
            "vitals_confidence is missing in this log, so unreliable CV readings cannot be filtered."
        )
    else:
        low_conf_ratio = _sample_value_ratio(
            data.confidence_samples,
            lambda value: value < 0.3,
        )
        avg_conf = sum(data.confidence_samples) / len(data.confidence_samples)
        if low_conf_ratio >= 0.25:
            notes.append(
                f"Low-confidence vitals frames are high ({low_conf_ratio:.0%}, average confidence {avg_conf:.2f})."
            )

    if hp_zero_ratio >= 0.25:
        notes.append(f"HP is near zero in {hp_zero_ratio:.0%} of frames.")
    if shield_zero_ratio >= 0.25:
        notes.append(f"Shield is near zero in {shield_zero_ratio:.0%} of frames.")

    if not notes:
        return '<div class="quality ok">Vitals readings look stable enough for timeline review.</div>'
    escaped = " ".join(html.escape(note) for note in notes)
    return f'<div class="quality">{escaped} ROI calibration or a newer session log is recommended.</div>'


def _ordered_actions(counter: Counter[str]) -> list[str]:
    base = ["NONE", "HEAL", "RETREAT", "TAKE_COVER", "TAKE_HIGH_GROUND", "PUSH"]
    extras = sorted(action for action in counter if action not in base)
    return [action for action in base if action in counter] + extras


@dataclass(frozen=True, slots=True)
class _PlotArea:
    left: float
    top: float
    right: float
    bottom: float

    def value_y(self, value: float) -> float:
        clipped = max(0.0, min(1.0, float(value)))
        return self.bottom - clipped * (self.bottom - self.top)

    def time_x(self, seconds: float, duration: float) -> float:
        if duration <= 0.0:
            return self.left
        return self.left + (max(0.0, float(seconds)) / duration) * (self.right - self.left)


def _svg_points(samples: list[tuple[float, float]], *, plot: _PlotArea) -> str:
    if not samples:
        return ""
    min_ts = samples[0][0]
    max_ts = samples[-1][0]
    span = max(1e-9, max_ts - min_ts)
    points: list[str] = []
    for ts, value in _downsample_samples(samples, limit=480):
        x = plot.left + ((ts - min_ts) / span) * (plot.right - plot.left)
        y = plot.value_y(value)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _time_ticks(duration: float) -> list[float]:
    duration = max(0.0, float(duration))
    if duration <= 0.0:
        return [0.0]
    raw_step = duration / 4.0
    if raw_step <= 1.0:
        step = 1.0
    elif raw_step <= 5.0:
        step = 5.0
    elif raw_step <= 15.0:
        step = 15.0
    elif raw_step <= 30.0:
        step = 30.0
    else:
        step = 60.0
    ticks = [0.0]
    value = step
    while value < duration:
        ticks.append(value)
        value += step
    if ticks[-1] != duration:
        ticks.append(duration)
    return ticks


def _format_tick_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    rest = int(round(seconds % 60))
    return f"{minutes}:{rest:02d}"


def _downsample_samples(
    samples: list[tuple[float, float]],
    *,
    limit: int,
) -> list[tuple[float, float]]:
    if len(samples) <= limit:
        return samples
    if limit <= 1:
        return [samples[0]]
    last = len(samples) - 1
    return [samples[round(i * last / (limit - 1))] for i in range(limit)]


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _sample_ratio(samples: list[tuple[float, float]], predicate) -> float:
    if not samples:
        return 0.0
    count = sum(1 for _, value in samples if predicate(value))
    return count / len(samples)


def _sample_value_ratio(samples: list[float], predicate) -> float:
    if not samples:
        return 0.0
    count = sum(1 for value in samples if predicate(value))
    return count / len(samples)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
