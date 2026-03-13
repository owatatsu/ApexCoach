from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional at runtime
    yaml = None


@dataclass(slots=True)
class Roi:
    x: int
    y: int
    w: int
    h: int


@dataclass(slots=True)
class FrequencyConfig:
    capture_fps: int = 10
    ui_parse_fps: int = 10
    ocr_fps: int = 2
    state_fps: int = 10
    decision_fps: int = 10
    llm_fps: int = 1
    overlay_fps: int = 10


@dataclass(slots=True)
class ThresholdConfig:
    low_total_hp_shield: float = 0.45
    heal_total_hp_shield: float = 0.65
    critical_heal_total_hp_shield: float = 0.35
    broken_shield_heal_hp_pct: float = 0.55
    push_min_total_hp_shield: float = 0.95
    high_damage_1s: float = 0.35
    high_damage_3s: float = 0.55
    min_damage_event_delta: float = 0.04
    quiet_damage_1s: float = 0.05
    ally_isolated_alive_count: int = 1
    knock_recent_seconds: float = 2.5
    under_fire_damage_1s: float = 0.03
    vitals_confidence_min: float = 0.3
    low_hp_consecutive_frames: int = 3
    heal_stationary_frames: int = 4
    retreat_lowhp_stationary_frames: int = 3
    movement_score_threshold: float = 0.045
    low_ground_confidence_min: float = 0.55
    exposed_confidence_min: float = 0.65


@dataclass(slots=True)
class ArbiterConfig:
    same_action_cooldown_seconds: float = 2.0
    retreat_hold_seconds: float = 1.2
    push_block_after_retreat_seconds: float = 3.0


@dataclass(slots=True)
class OverlayConfig:
    enabled: bool = True
    show_window: bool = False
    show_reason: bool = True
    window_name: str = "ApexCoach"
    window_mode: str = "hud"  # hud | frame
    window_click_through: bool = True
    window_transparent: bool = True
    window_always_on_top: bool = True
    display_hold_seconds: float = 3.0
    max_lines: int = 3
    text_scale: float = 0.85
    background_alpha: float = 0.5
    position: str = "right_center"
    margin_x: int = 36
    offset_y: int = 0
    panel_width: int = 540
    text_x: int = 36
    text_y: int = 54
    debug_show_rois: bool = False
    debug_roi_alpha: float = 0.35
    debug_show_roi_labels: bool = True


@dataclass(slots=True)
class OfflineConfig:
    input_video: str = ""
    output_video: str = ""
    telemetry_jsonl: str = ""


@dataclass(slots=True)
class RealtimeConfig:
    monitor_index: int = 1
    region_x: int = 0
    region_y: int = 0
    region_w: int = 0
    region_h: int = 0
    duration_seconds: float = 0.0


@dataclass(slots=True)
class LoggingConfig:
    enabled: bool = True
    path: str = "logs/session_{timestamp}.jsonl"
    include_reason: bool = True


@dataclass(slots=True)
class LlmConfig:
    enabled: bool = False
    provider: str = "lmstudio"
    model: str = ""
    model_name: str = ""
    model_names: list[str] = field(default_factory=list)
    offline_review_model_name: str = ""
    offline_review_model_names: list[str] = field(default_factory=list)
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "lm-studio"
    timeout_seconds: float = 45.0  # offline review path
    timeout_ms: int = 300  # realtime advisor path
    temperature: float = 0.1
    num_ctx: int = 4096
    lmstudio_response_format: str = "json_schema"
    llm_max_tokens: int = 64
    max_reason_chars: int = 48
    min_request_interval_ms: int = 1000
    parse_retry_count: int = 1
    failure_threshold: int = 5
    disable_seconds: float = 10.0
    request_log_path: str = "logs/llm_requests.jsonl"
    max_raw_response_chars: int = 1200
    mock_return_none: bool = False
    mock_delay_ms: int = 0
    advice_enabled: bool = True
    frame_reasoning_enabled: bool = False
    offline_review_enabled: bool = True
    offline_review_output: str = "logs/coach_review_{timestamp}.md"
    offline_review_max_events: int = 16
    offline_review_max_tokens: int = 1024
    offline_review_prompt_max_chars: int = 12000
    offline_review_reason_max_chars: int = 96
    offline_review_language: str = "ja"


@dataclass(slots=True)
class PerformanceConfig:
    parallel_io: bool = True
    read_prefetch_queue_size: int = 64
    write_queue_size: int = 64
    opencv_threads: int = 0


def default_rois() -> dict[str, Roi]:
    return {
        # Baseline for 1920x1080 (Apex HUD with teammate panel on left-middle).
        # Use overlay.debug_show_rois=true to visually fine tune per resolution/UI scale.
        "hp_bar": Roi(42, 1006, 420, 24),
        "shield_bar": Roi(42, 982, 420, 22),
        "teammate_panel": Roi(20, 735, 430, 220),
        "kill_feed": Roi(1490, 120, 390, 220),
    }


@dataclass(slots=True)
class ApexCoachConfig:
    frequencies: FrequencyConfig = field(default_factory=FrequencyConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    rois: dict[str, Roi] = field(default_factory=default_rois)
    scale_rois_to_frame: bool = True
    roi_reference_width: int = 1920
    roi_reference_height: int = 1080


def load_config(path: str | Path | None) -> ApexCoachConfig:
    config = ApexCoachConfig()
    if path is None:
        return config

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    raw = path_obj.read_text(encoding="utf-8")
    suffix = path_obj.suffix.lower()
    if suffix == ".json":
        parsed = json.loads(raw)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files.")
        parsed = yaml.safe_load(raw) or {}

    if not isinstance(parsed, dict):
        raise ValueError("Top-level config must be a dictionary.")

    _merge_dataclass(config, parsed)
    return config


def format_run_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y%m%d_%H%M%S")


def _merge_dataclass(target: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(target, key):
            continue

        if key == "rois" and isinstance(value, dict):
            merged: dict[str, Roi] = {}
            for roi_name, roi_value in value.items():
                if isinstance(roi_value, dict):
                    merged[roi_name] = Roi(
                        x=int(roi_value["x"]),
                        y=int(roi_value["y"]),
                        w=int(roi_value["w"]),
                        h=int(roi_value["h"]),
                    )
                elif isinstance(roi_value, list) and len(roi_value) == 4:
                    merged[roi_name] = Roi(
                        x=int(roi_value[0]),
                        y=int(roi_value[1]),
                        w=int(roi_value[2]),
                        h=int(roi_value[3]),
                    )
            setattr(target, key, merged)
            continue

        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
            continue

        setattr(target, key, value)
