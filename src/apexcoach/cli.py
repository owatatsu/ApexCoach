from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apexcoach.aim_diagnosis import AimDiagnosisService, ClipRange
from apexcoach.config import ApexCoachConfig, load_config
from apexcoach.pipeline import OfflinePipeline, RealtimePipeline

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None

from apexcoach.capture_service import probe_video_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ApexCoach MVP (rule-based)")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")
    parser.add_argument(
        "--aim-diagnosis",
        action="store_true",
        help="Run recording-based aim diagnosis instead of the tactical pipeline.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Use realtime screen capture instead of offline video input",
    )
    parser.add_argument("--video", type=str, default=None, help="Offline input video path")
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="Realtime monitor index for capture (mss monitor index)",
    )
    parser.add_argument(
        "--region",
        type=_parse_region,
        default=None,
        help="Realtime capture region: x,y,w,h (screen coordinates)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Realtime capture duration in seconds (0 or omitted = until Ctrl+C)",
    )
    parser.add_argument(
        "--telemetry",
        type=str,
        default=None,
        help="Optional telemetry JSONL path",
    )
    parser.add_argument("--log", type=str, default=None, help="Session log JSONL path")
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output video path with overlay",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show overlay preview window while running",
    )
    parser.add_argument(
        "--disable-overlay",
        action="store_true",
        help="Disable overlay rendering",
    )
    parser.add_argument(
        "--llm-enable",
        action="store_true",
        help="Enable local LLM features (offline review).",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="Local LLM provider: lmstudio or ollama.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Local LLM model name.",
    )
    parser.add_argument(
        "--llm-models",
        type=str,
        default=None,
        help="Comma-separated LLM model names to try for realtime advice.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Local LLM API base URL (example: http://localhost:1234/v1).",
    )
    parser.add_argument(
        "--llm-review-model",
        type=str,
        default=None,
        help="Override model name for offline review only.",
    )
    parser.add_argument(
        "--llm-review-models",
        type=str,
        default=None,
        help="Comma-separated model names to try for offline review.",
    )
    parser.add_argument(
        "--llm-review-output",
        type=str,
        default=None,
        help="Output path for offline review markdown.",
    )
    parser.add_argument(
        "--clips-json",
        type=str,
        default=None,
        help="Clip definition JSON for aim diagnosis mode.",
    )
    parser.add_argument(
        "--aim-output",
        type=str,
        default=None,
        help="Output path for aim diagnosis result JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.aim_diagnosis:
        _validate_aim_diagnosis_inputs(args, parser)
        result = _run_aim_diagnosis(args)
        print("Aim diagnosis finished.")
        print(
            f"video_id={result['video_id']} "
            f"clips={len(result['clips'])} "
            f"priority_labels={','.join(result['training_plan']['priority_labels']) if result['training_plan'] else 'none'}"
        )
        if args.aim_output:
            print(f"saved={Path(args.aim_output).expanduser().resolve()}")
        return

    config = load_config(args.config)
    _apply_cli_overrides(config, args)
    _validate_inputs(config, parser, realtime_mode=args.realtime)

    if args.realtime:
        summary = RealtimePipeline(config).run()
    else:
        summary = OfflinePipeline(config).run()
    print("ApexCoach run finished.")
    print(
        f"frames={summary['frames']} "
        f"NONE={summary['NONE']} "
        f"HEAL={summary['HEAL']} "
        f"RETREAT={summary['RETREAT']} "
        f"TAKE_COVER={summary['TAKE_COVER']} "
        f"TAKE_HIGH_GROUND={summary['TAKE_HIGH_GROUND']} "
        f"PUSH={summary['PUSH']}"
    )


def _apply_cli_overrides(config: ApexCoachConfig, args: argparse.Namespace) -> None:
    if args.realtime and not args.show_window and not args.disable_overlay:
        # Realtime mode should be visible by default.
        config.overlay.show_window = True
    if args.video:
        config.offline.input_video = args.video
    if args.telemetry:
        config.offline.telemetry_jsonl = args.telemetry
    if args.output_video:
        config.offline.output_video = args.output_video
    if args.log:
        config.logging.path = args.log
    if args.monitor is not None:
        config.realtime.monitor_index = int(args.monitor)
    if args.region:
        x, y, w, h = args.region
        config.realtime.region_x = x
        config.realtime.region_y = y
        config.realtime.region_w = w
        config.realtime.region_h = h
    if args.duration is not None:
        config.realtime.duration_seconds = max(0.0, float(args.duration))
    if args.show_window:
        config.overlay.show_window = True
    if args.disable_overlay:
        config.overlay.enabled = False
    if args.llm_enable:
        config.llm.enabled = True
    if args.llm_provider:
        config.llm.provider = args.llm_provider
    if args.llm_model:
        config.llm.model = args.llm_model
        config.llm.model_name = args.llm_model
        config.llm.model_names = [args.llm_model]
    if args.llm_models:
        model_names = _parse_csv_models(args.llm_models)
        config.llm.model_names = model_names
        if model_names:
            config.llm.model = model_names[0]
            config.llm.model_name = model_names[0]
    if args.llm_review_model:
        config.llm.offline_review_model_name = args.llm_review_model
        config.llm.offline_review_model_names = [args.llm_review_model]
    if args.llm_review_models:
        review_model_names = _parse_csv_models(args.llm_review_models)
        config.llm.offline_review_model_names = review_model_names
        if review_model_names:
            config.llm.offline_review_model_name = review_model_names[0]
    if args.llm_base_url:
        config.llm.base_url = args.llm_base_url
    if args.llm_review_output:
        config.llm.offline_review_output = args.llm_review_output


def _validate_inputs(
    config: ApexCoachConfig, parser: argparse.ArgumentParser, realtime_mode: bool
) -> None:
    if config.llm.enabled:
        provider = (config.llm.provider or "").strip().lower()
        allowed = {"lmstudio", "mock", "ollama"}
        if provider not in allowed:
            parser.error(
                f"Unsupported llm.provider='{config.llm.provider}'. "
                f"Supported values: {', '.join(sorted(allowed))}."
            )

    if realtime_mode:
        if config.realtime.monitor_index <= 0:
            parser.error("--monitor must be >= 1 for realtime mode.")
        return

    video_path = Path(config.offline.input_video).expanduser()
    if not config.offline.input_video:
        parser.error("--video is required for offline mode.")
    if not video_path.exists():
        parser.error(
            "Video file does not exist: "
            f"{video_path.resolve()}. "
            "Replace placeholder paths like 'path\\to\\apex_recording.mp4' with your real file path."
        )

    if config.offline.telemetry_jsonl:
        telemetry_path = Path(config.offline.telemetry_jsonl).expanduser()
        if not telemetry_path.exists():
            parser.error(
                f"Telemetry file does not exist: {telemetry_path.resolve()}"
            )


def _parse_region(region: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in region.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--region must be in format x,y,w,h (4 integers)."
        )
    try:
        x, y, w, h = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--region must contain integers only."
        ) from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(
            "--region width and height must be > 0."
        )
    return x, y, w, h


def _validate_aim_diagnosis_inputs(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.realtime:
        parser.error("--realtime cannot be used with --aim-diagnosis.")
    if not args.video:
        parser.error("--video is required for aim diagnosis mode.")
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        parser.error(f"Video file does not exist: {video_path.resolve()}")
    if not args.clips_json:
        parser.error("--clips-json is required for aim diagnosis mode.")
    clips_path = Path(args.clips_json).expanduser()
    if not clips_path.exists():
        parser.error(f"Clip JSON does not exist: {clips_path.resolve()}")


def _run_aim_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    video_path = Path(args.video).expanduser().resolve()
    video_id = video_path.stem
    clips = _load_clip_ranges(Path(args.clips_json), video_id=video_id)

    service = AimDiagnosisService()
    service.register_video(
        file_path=str(video_path),
        duration_sec=_probe_video_duration_seconds(video_path),
        video_id=video_id,
    )
    service.save_clips(video_id=video_id, clips=clips)
    service.start_analysis(video_id=video_id)
    service.analyze_saved_clips(video_id=video_id)
    result = service.get_results(video_id=video_id)

    if args.aim_output:
        _write_aim_result(Path(args.aim_output), result)
    return result


def _load_clip_ranges(path: Path, *, video_id: str) -> list[ClipRange]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_clips = payload.get("clips")
    else:
        raw_clips = payload
    if not isinstance(raw_clips, list):
        raise ValueError("clips.json must contain a list or a {'clips': [...]} object.")

    clips: list[ClipRange] = []
    for index, raw_clip in enumerate(raw_clips, start=1):
        if not isinstance(raw_clip, dict):
            raise ValueError("Each clip entry must be an object.")
        clip_id = str(raw_clip.get("clip_id") or raw_clip.get("id") or f"clip_{index:03d}")
        start_sec = float(raw_clip["start_sec"])
        end_sec = float(raw_clip["end_sec"])
        note = str(raw_clip.get("note") or "")
        clips.append(
            ClipRange(
                id=clip_id,
                video_id=video_id,
                start_sec=start_sec,
                end_sec=end_sec,
                note=note,
            )
        )
    return clips


def _write_aim_result(path: Path, result: dict[str, Any]) -> None:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _probe_video_duration_seconds(video_path: Path) -> float:
    metadata = probe_video_metadata(video_path)
    if metadata.duration_sec > 0.0:
        return metadata.duration_sec
    if cv2 is None:
        return 0.0
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0.0:
        return 0.0
    return max(0.0, frame_count / fps)


def _parse_csv_models(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]
