from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AimLabel(str, Enum):
    SLOW_INITIAL_ADJUSTMENT = "slow_initial_adjustment"
    OVERFLICK = "overflick"
    TRACKING_DELAY = "tracking_delay"
    RECOIL_BREAKDOWN = "recoil_breakdown"
    CLOSE_RANGE_INSTABILITY = "close_range_instability"
    ADS_JUDGMENT_ISSUE = "ads_judgment_issue"


@dataclass(slots=True)
class VideoRef:
    id: str
    file_path: str
    duration_sec: float
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class ClipRange:
    id: str
    video_id: str
    start_sec: float
    end_sec: float
    note: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(slots=True)
class ClipAnalysisMetrics:
    time_to_first_shot_ms: int | None = None
    aim_path_variance: float | None = None
    tracking_error_score: float | None = None
    recoil_error_score: float | None = None
    close_range_score: float | None = None
    ads_usage_score: float | None = None


@dataclass(slots=True)
class ClipDiagnosis:
    clip_id: str
    summary: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    labels: list[AimLabel] = field(default_factory=list)
    confidence: float = 0.0
    next_focus: str = ""
    metrics: ClipAnalysisMetrics = field(default_factory=ClipAnalysisMetrics)


@dataclass(slots=True)
class TrainingDrill:
    label: AimLabel
    title: str
    minutes: int
    instruction: str
    check_point: str


@dataclass(slots=True)
class TrainingPlan:
    video_id: str
    priority_labels: list[AimLabel] = field(default_factory=list)
    drills: list[TrainingDrill] = field(default_factory=list)
    total_minutes: int = 0

