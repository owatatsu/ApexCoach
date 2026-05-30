import json

from apexcoach.session_report import build_session_report, render_session_report, write_session_report


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
