from __future__ import annotations

import argparse
from pathlib import Path

from apexcoach.config import ApexCoachConfig, load_config
from apexcoach.pipeline import OfflinePipeline, RealtimePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ApexCoach MVP (rule-based)")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")
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
        "--llm-base-url",
        type=str,
        default=None,
        help="Local LLM API base URL (example: http://localhost:1234/v1).",
    )
    parser.add_argument(
        "--llm-review-output",
        type=str,
        default=None,
        help="Output path for offline review markdown.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

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
