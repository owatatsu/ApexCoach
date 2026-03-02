from __future__ import annotations

import argparse
from pathlib import Path

from apexcoach.config import ApexCoachConfig, load_config
from apexcoach.pipeline import OfflinePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ApexCoach MVP (rule-based)")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")
    parser.add_argument("--video", type=str, default=None, help="Offline input video path")
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    _apply_cli_overrides(config, args)
    _validate_inputs(config, parser)

    summary = OfflinePipeline(config).run()
    print("ApexCoach run finished.")
    print(
        f"frames={summary['frames']} "
        f"NONE={summary['NONE']} "
        f"HEAL={summary['HEAL']} "
        f"RETREAT={summary['RETREAT']} "
        f"PUSH={summary['PUSH']}"
    )


def _apply_cli_overrides(config: ApexCoachConfig, args: argparse.Namespace) -> None:
    if args.video:
        config.offline.input_video = args.video
    if args.telemetry:
        config.offline.telemetry_jsonl = args.telemetry
    if args.output_video:
        config.offline.output_video = args.output_video
    if args.log:
        config.logging.path = args.log
    if args.show_window:
        config.overlay.show_window = True
    if args.disable_overlay:
        config.overlay.enabled = False


def _validate_inputs(config: ApexCoachConfig, parser: argparse.ArgumentParser) -> None:
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
