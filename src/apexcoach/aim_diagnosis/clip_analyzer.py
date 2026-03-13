from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from apexcoach.aim_diagnosis.models import ClipAnalysisMetrics, ClipRange
from apexcoach.capture_service import VideoCaptureService

if TYPE_CHECKING:
    import numpy as np

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None
    np = None


@dataclass(slots=True)
class ClipAnalyzerConfig:
    sample_fps: int = 12
    motion_threshold: int = 20
    center_region: tuple[float, float, float, float] = (0.3, 0.2, 0.7, 0.8)
    ads_core_region: tuple[float, float, float, float] = (0.42, 0.32, 0.58, 0.68)
    min_frames: int = 4


def analyze_clip(
    *,
    video_path: str | Path,
    clip: ClipRange,
    config: ClipAnalyzerConfig | None = None,
) -> ClipAnalysisMetrics:
    config = config or ClipAnalyzerConfig()
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for clip analysis.")

    timestamps: list[float] = []
    motion_strengths: list[float] = []
    coverages: list[float] = []
    core_ratios: list[float] = []
    centroids: list[tuple[float, float]] = []
    recoil_values: list[float] = []
    first_motion_ts: float | None = None
    prev_gray = None
    prev_centroid_y: float | None = None

    with VideoCaptureService(video_path=video_path, target_fps=config.sample_fps) as capture:
        for packet in capture.iter_frames():
            if packet.timestamp + 1e-9 < clip.start_sec:
                continue
            if packet.timestamp > clip.end_sec + 1e-9:
                break

            region = _crop_region(packet.frame, config.center_region)
            if region.size == 0:
                continue

            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            timestamps.append(packet.timestamp)

            if prev_gray is None:
                prev_gray = gray
                continue

            diff = cv2.absdiff(gray, prev_gray)
            prev_gray = gray

            motion_mask = (diff >= int(config.motion_threshold)).astype(np.uint8)
            motion_strength = float(diff.mean() / 255.0)
            coverage = float(motion_mask.mean())

            if first_motion_ts is None and (motion_strength >= 0.03 or coverage >= 0.015):
                first_motion_ts = packet.timestamp

            centroid = _motion_centroid(motion_mask)
            centroids.append(centroid)
            motion_strengths.append(motion_strength)
            coverages.append(coverage)
            core_ratios.append(_core_ratio(motion_mask, config.ads_core_region))

            if prev_centroid_y is not None:
                recoil_values.append(abs(centroid[1] - prev_centroid_y))
            prev_centroid_y = centroid[1]

    if len(motion_strengths) < max(1, int(config.min_frames) - 1):
        return ClipAnalysisMetrics()

    return ClipAnalysisMetrics(
        time_to_first_shot_ms=_time_to_first_motion_ms(clip, first_motion_ts),
        aim_path_variance=round(_aim_path_variance(centroids), 3),
        tracking_error_score=round(_tracking_error_score(centroids), 3),
        recoil_error_score=round(_recoil_error_score(recoil_values, motion_strengths), 3),
        close_range_score=round(_close_range_score(coverages, motion_strengths), 3),
        ads_usage_score=round(_ads_usage_score(core_ratios), 3),
    )


def analyze_clips(
    *,
    video_path: str | Path,
    clips: Sequence[ClipRange],
    config: ClipAnalyzerConfig | None = None,
) -> dict[str, ClipAnalysisMetrics]:
    config = config or ClipAnalyzerConfig()
    metrics_by_clip: dict[str, ClipAnalysisMetrics] = {}
    for clip in clips:
        metrics_by_clip[clip.id] = analyze_clip(
            video_path=video_path,
            clip=clip,
            config=config,
        )
    return metrics_by_clip


def _crop_region(frame: "np.ndarray", region: tuple[float, float, float, float]) -> "np.ndarray":
    height, width = frame.shape[:2]
    x1 = int(round(width * region[0]))
    y1 = int(round(height * region[1]))
    x2 = int(round(width * region[2]))
    y2 = int(round(height * region[3]))
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return frame[0:0, 0:0]
    return frame[y1:y2, x1:x2]


def _motion_centroid(mask: "np.ndarray") -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape[:2]
    if len(xs) == 0 or width <= 1 or height <= 1:
        return 0.5, 0.5
    return (
        float(xs.mean()) / float(width - 1),
        float(ys.mean()) / float(height - 1),
    )


def _core_ratio(mask: "np.ndarray", ads_core_region: tuple[float, float, float, float]) -> float:
    total = float(mask.mean())
    if total <= 0.0:
        return 0.0
    core = _crop_region(mask, ads_core_region)
    if core.size == 0:
        return 0.0
    core_mean = float(core.mean())
    return max(0.0, min(1.0, core_mean / total))


def _time_to_first_motion_ms(clip: ClipRange, first_motion_ts: float | None) -> int | None:
    if first_motion_ts is None:
        return None
    delay_ms = max(0.0, (float(first_motion_ts) - float(clip.start_sec)) * 1000.0)
    return int(round(delay_ms))


def _aim_path_variance(centroids: Sequence[tuple[float, float]]) -> float:
    if len(centroids) < 2:
        return 0.0
    offsets = [((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5 for x, y in centroids]
    return _clamp01(float(np.std(offsets)) * 4.0)


def _tracking_error_score(centroids: Sequence[tuple[float, float]]) -> float:
    if len(centroids) < 3:
        return 0.0
    xs = np.array([x for x, _ in centroids], dtype=float)
    velocities = np.diff(xs)
    if velocities.size < 2:
        return 0.0
    accelerations = np.diff(velocities)
    return _clamp01(float(np.mean(np.abs(accelerations))) * 10.0)


def _recoil_error_score(
    recoil_values: Sequence[float],
    motion_strengths: Sequence[float],
) -> float:
    if not recoil_values:
        return 0.0
    baseline = float(np.mean(motion_strengths)) if motion_strengths else 0.0
    return _clamp01(float(np.mean(recoil_values)) * 6.0 + baseline * 0.8)


def _close_range_score(
    coverages: Sequence[float],
    motion_strengths: Sequence[float],
) -> float:
    if not coverages:
        return 0.0
    coverage_score = float(np.mean(coverages)) * 16.0
    motion_bonus = float(np.mean(motion_strengths)) * 3.5 if motion_strengths else 0.0
    return _clamp01(coverage_score + motion_bonus)


def _ads_usage_score(core_ratios: Sequence[float]) -> float:
    if not core_ratios:
        return 0.0
    return _clamp01(float(np.mean(core_ratios)) * 0.9)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
