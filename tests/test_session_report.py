import json
from threading import Event, Thread

from apexcoach.models import Action, ArbiterResult, Decision, FrameEvents, FramePacket, GameState
from apexcoach.session_report import build_session_report, render_session_report, write_session_report
from apexcoach.session_logger import SessionLogger


def test_build_session_report_extracts_counts_and_events(tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    records = [
        {
            "timestamp": 0.0,
            "state": {
                "hp_pct": 1.0,
                "shield_pct": 1.0,
                "under_fire": False,
                "allies_alive": 3,
                "allies_down": 0,
            },
            "decision": {"action": "NONE", "reason": "No strong signal."},
            "arbiter": {"action": "NONE", "emitted": False},
        },
        {
            "timestamp": 1.0,
            "state": {
                "hp_pct": 0.25,
                "shield_pct": 0.2,
                "under_fire": True,
                "allies_alive": 2,
                "allies_down": 1,
            },
            "decision": {"action": "RETREAT", "reason": "High incoming damage in last 1s."},
            "arbiter": {"action": "RETREAT", "emitted": True},
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = build_session_report(log_path)

    assert report.frames == 2
    assert report.duration_sec == 1.0
    assert report.action_counts["RETREAT"] == 1
    assert report.emitted_counts["RETREAT"] == 1
    assert report.under_fire_frames == 1
    assert report.low_resource_frames == 1
    assert len(report.events) == 1


def test_render_session_report_escapes_content(tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": 0.0,
                "state": {"hp_pct": 0.4, "shield_pct": 0.1, "under_fire": True},
                "decision": {"action": "HEAL", "reason": "<heal now>"},
                "arbiter": {"action": "HEAL", "emitted": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    html = render_session_report(build_session_report(log_path))

    assert "ApexCoach Session Report" in html
    assert "&lt;heal now&gt;" in html
    assert "Vitals Timeline" in html


def test_write_session_report_creates_html(tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    output_path = tmp_path / "report.html"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": 0.0,
                "state": {"hp_pct": 1.0, "shield_pct": 1.0},
                "decision": {"action": "NONE", "reason": ""},
                "arbiter": {"action": "NONE", "emitted": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    saved = write_session_report(log_path=log_path, output_path=output_path)

    assert saved == str(output_path)
    assert output_path.exists()
    assert "<!doctype html>" in output_path.read_text(encoding="utf-8")


def test_session_report_extracts_voice_metrics_without_counting_voice_as_frame(tmp_path) -> None:
    log_path = tmp_path / "session.jsonl"
    records = [
        {
            "timestamp": 0.0,
            "state": {"hp_pct": 1.0, "shield_pct": 1.0},
            "decision": {"action": "NONE", "reason": ""},
            "arbiter": {"action": "NONE", "emitted": False},
        },
        {
            "record_type": "voice_event",
            "event": "started",
            "action": "RETREAT",
            "latency_ms": 120.0,
        },
        {
            "record_type": "voice_event",
            "event": "completed",
            "action": "RETREAT",
            "latency_ms": 120.0,
        },
        {
            "record_type": "voice_event",
            "event": "dropped",
            "action": "PUSH",
            "dropped_reason": "expired",
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = build_session_report(log_path)

    assert report.frames == 1
    assert report.voice_success_count == 1
    assert report.voice_dropped_count == 1
    assert report.voice_p95_start_latency_ms == 120.0


def test_session_logger_close_race_keeps_jsonl_valid(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    logger = SessionLogger(str(path), enabled=True)
    packet = FramePacket(frame_index=1, timestamp=1.0, frame=None)
    state = GameState(timestamp=1.0)
    events = FrameEvents(timestamp=1.0)
    decision = Decision(Action.NONE, "none")
    arbiter = ArbiterResult(Action.NONE, False, "none", Action.NONE)
    started = Event()

    def write_events() -> None:
        for index in range(20):
            logger.log_voice_event(
                {"event": "enqueued", "action": "PUSH", "queue_depth": index}
            )
            if index == 0:
                started.set()
            logger.log_frame(
                packet,
                state,
                events,
                decision,
                arbiter,
            )

    writer = Thread(target=write_events)
    writer.start()
    assert started.wait(timeout=1.0)
    logger.close()
    writer.join(timeout=2.0)
    logger.log_voice_event({"event": "completed", "action": "PUSH"})

    assert not writer.is_alive()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines
    assert all(isinstance(json.loads(line), dict) for line in lines)
