import argparse
import json
from types import SimpleNamespace

from apexcoach.cli import (
    _apply_cli_overrides,
    _load_clip_ranges,
    _parse_region,
    _run_aim_diagnosis,
    _validate_aim_diagnosis_inputs,
    _validate_inputs,
    build_parser,
)
from apexcoach.config import ApexCoachConfig


def test_parse_region_ok() -> None:
    assert _parse_region("10,20,300,400") == (10, 20, 300, 400)


def test_parse_region_invalid() -> None:
    try:
        _parse_region("10,20,300")
        assert False, "expected ArgumentTypeError"
    except argparse.ArgumentTypeError:
        pass


def test_realtime_defaults_show_window() -> None:
    cfg = ApexCoachConfig()
    args = SimpleNamespace(
        realtime=True,
        show_window=False,
        disable_overlay=False,
        video=None,
        telemetry=None,
        output_video=None,
        log=None,
        monitor=None,
        region=None,
        duration=None,
        llm_enable=False,
        llm_provider=None,
        llm_model=None,
        llm_models=None,
        llm_review_model=None,
        llm_review_models=None,
        llm_base_url=None,
        llm_review_output=None,
    )
    _apply_cli_overrides(cfg, args)
    assert cfg.overlay.show_window is True


def test_llm_cli_overrides() -> None:
    cfg = ApexCoachConfig()
    args = SimpleNamespace(
        realtime=False,
        show_window=False,
        disable_overlay=False,
        video=None,
        telemetry=None,
        output_video=None,
        log=None,
        monitor=None,
        region=None,
        duration=None,
        llm_enable=True,
        llm_provider="lmstudio",
        llm_model="qwen2.5-14b-instruct-q4_k_m.gguf",
        llm_models=None,
        llm_review_model="qwen3.5-9b",
        llm_review_models=None,
        llm_base_url="http://127.0.0.1:1234",
        llm_review_output="logs/custom_review.md",
    )
    _apply_cli_overrides(cfg, args)
    assert cfg.llm.enabled is True
    assert cfg.llm.provider == "lmstudio"
    assert cfg.llm.model == "qwen2.5-14b-instruct-q4_k_m.gguf"
    assert cfg.llm.model_name == "qwen2.5-14b-instruct-q4_k_m.gguf"
    assert cfg.llm.model_names == ["qwen2.5-14b-instruct-q4_k_m.gguf"]
    assert cfg.llm.offline_review_model_name == "qwen3.5-9b"
    assert cfg.llm.offline_review_model_names == ["qwen3.5-9b"]
    assert cfg.llm.base_url == "http://127.0.0.1:1234"
    assert cfg.llm.offline_review_output == "logs/custom_review.md"


def test_validate_inputs_rejects_invalid_llm_provider() -> None:
    cfg = ApexCoachConfig()
    cfg.llm.enabled = True
    cfg.llm.provider = "invalid-provider"
    parser = build_parser()
    try:
        _validate_inputs(cfg, parser, realtime_mode=True)
        assert False, "expected parser error"
    except SystemExit:
        pass


def test_parser_accepts_aim_diagnosis_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--aim-diagnosis",
            "--video",
            "match.mp4",
            "--clips-json",
            "clips.json",
            "--llm-models",
            "gpt-oss-swallow-20b,qwen3.5-9b",
            "--llm-review-model",
            "qwen3.5-9b",
            "--llm-review-models",
            "qwen3.5-9b,qwen2.5-14b",
            "--aim-output",
            "out/result.json",
        ]
    )
    assert args.aim_diagnosis is True
    assert args.video == "match.mp4"
    assert args.clips_json == "clips.json"
    assert args.llm_models == "gpt-oss-swallow-20b,qwen3.5-9b"
    assert args.llm_review_model == "qwen3.5-9b"
    assert args.llm_review_models == "qwen3.5-9b,qwen2.5-14b"
    assert args.aim_output == "out/result.json"


def test_llm_cli_plural_model_overrides() -> None:
    cfg = ApexCoachConfig()
    args = SimpleNamespace(
        realtime=False,
        show_window=False,
        disable_overlay=False,
        video=None,
        telemetry=None,
        output_video=None,
        log=None,
        monitor=None,
        region=None,
        duration=None,
        llm_enable=True,
        llm_provider="lmstudio",
        llm_model=None,
        llm_models="gpt-oss-swallow-20b,qwen3.5-9b",
        llm_review_model=None,
        llm_review_models="qwen3.5-9b,qwen2.5-14b",
        llm_base_url=None,
        llm_review_output=None,
    )

    _apply_cli_overrides(cfg, args)

    assert cfg.llm.model_names == ["gpt-oss-swallow-20b", "qwen3.5-9b"]
    assert cfg.llm.model_name == "gpt-oss-swallow-20b"
    assert cfg.llm.offline_review_model_names == ["qwen3.5-9b", "qwen2.5-14b"]
    assert cfg.llm.offline_review_model_name == "qwen3.5-9b"


def test_load_clip_ranges_from_object_payload(tmp_path) -> None:
    clips_path = tmp_path / "clips.json"
    clips_path.write_text(
        json.dumps(
            {
                "clips": [
                    {"start_sec": 12.5, "end_sec": 18.0, "note": "entry"},
                    {"clip_id": "fight_b", "start_sec": 34.0, "end_sec": 40.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    clips = _load_clip_ranges(clips_path, video_id="vid_1")

    assert [clip.id for clip in clips] == ["clip_001", "fight_b"]
    assert [clip.video_id for clip in clips] == ["vid_1", "vid_1"]
    assert clips[0].note == "entry"


def test_validate_aim_diagnosis_inputs_rejects_realtime(tmp_path) -> None:
    video_path = tmp_path / "match.mp4"
    clips_path = tmp_path / "clips.json"
    video_path.write_bytes(b"not-a-real-video")
    clips_path.write_text("[]", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            "--aim-diagnosis",
            "--realtime",
            "--video",
            str(video_path),
            "--clips-json",
            str(clips_path),
        ]
    )

    try:
        _validate_aim_diagnosis_inputs(args, parser)
        assert False, "expected parser error"
    except SystemExit:
        pass


def test_run_aim_diagnosis_writes_output(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "match.mp4"
    clips_path = tmp_path / "clips.json"
    output_path = tmp_path / "results" / "aim_result.json"
    video_path.write_bytes(b"placeholder")
    clips_path.write_text(
        json.dumps([{"clip_id": "clip_1", "start_sec": 1.0, "end_sec": 4.0}]),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self) -> None:
            self.video_id = ""

        def register_video(self, *, file_path: str, duration_sec: float, video_id: str):
            self.video_id = video_id

        def save_clips(self, *, video_id: str, clips):
            assert video_id == self.video_id
            assert len(clips) == 1

        def start_analysis(self, *, video_id: str):
            assert video_id == self.video_id

        def analyze_saved_clips(self, *, video_id: str):
            assert video_id == self.video_id
            return []

        def get_results(self, *, video_id: str):
            assert video_id == self.video_id
            return {
                "video_id": video_id,
                "clips": [{"clip_id": "clip_1"}],
                "training_plan": {"priority_labels": ["tracking_delay"]},
            }

    monkeypatch.setattr("apexcoach.cli.AimDiagnosisService", FakeService)
    monkeypatch.setattr("apexcoach.cli._probe_video_duration_seconds", lambda path: 42.0)

    args = build_parser().parse_args(
        [
            "--aim-diagnosis",
            "--video",
            str(video_path),
            "--clips-json",
            str(clips_path),
            "--aim-output",
            str(output_path),
        ]
    )

    result = _run_aim_diagnosis(args)

    assert result["video_id"] == "match"
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["training_plan"]["priority_labels"] == ["tracking_delay"]
