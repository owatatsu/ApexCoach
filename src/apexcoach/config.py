from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
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
    push_min_total_hp_shield: float = 0.95
    high_damage_1s: float = 0.35
    high_damage_3s: float = 0.55
    quiet_damage_1s: float = 0.05
    ally_isolated_alive_count: int = 1
    knock_recent_seconds: float = 2.5
    under_fire_damage_1s: float = 0.03


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
    position: str = "right_center"
    margin_x: int = 36
    offset_y: int = 0
    panel_width: int = 540
    text_x: int = 36
    text_y: int = 54


@dataclass(slots=True)
class OfflineConfig:
    input_video: str = ""
    output_video: str = ""
    telemetry_jsonl: str = ""


@dataclass(slots=True)
class LoggingConfig:
    enabled: bool = True
    path: str = "logs/session.jsonl"
    include_reason: bool = True


@dataclass(slots=True)
class LlmConfig:
    enabled: bool = False


def default_rois() -> dict[str, Roi]:
    return {
        "hp_bar": Roi(136, 991, 294, 22),
        "shield_bar": Roi(136, 967, 294, 22),
        "teammate_panel": Roi(1650, 850, 240, 210),
        "kill_feed": Roi(1490, 120, 390, 220),
    }


@dataclass(slots=True)
class ApexCoachConfig:
    frequencies: FrequencyConfig = field(default_factory=FrequencyConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    rois: dict[str, Roi] = field(default_factory=default_rois)


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
