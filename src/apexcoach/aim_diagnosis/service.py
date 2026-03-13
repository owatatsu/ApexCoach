from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import uuid4

from apexcoach.aim_diagnosis.clip_analyzer import ClipAnalyzerConfig, analyze_clips
from apexcoach.aim_diagnosis.comment_builder import build_clip_diagnoses
from apexcoach.aim_diagnosis.labeler import infer_labels
from apexcoach.aim_diagnosis.models import (
    AimLabel,
    ClipDiagnosis,
    ClipRange,
    ClipAnalysisMetrics,
    TrainingPlan,
    VideoRef,
)
from apexcoach.aim_diagnosis.training_planner import build_training_plan


@dataclass(slots=True)
class AnalysisJob:
    id: str
    video_id: str
    status: str = "queued"
    requested_labels: list[AimLabel] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AimDiagnosisService:
    def __init__(self) -> None:
        self._videos: dict[str, VideoRef] = {}
        self._clips_by_video: dict[str, list[ClipRange]] = {}
        self._jobs_by_video: dict[str, AnalysisJob] = {}
        self._diagnoses_by_video: dict[str, dict[str, ClipDiagnosis]] = {}
        self._training_by_video: dict[str, TrainingPlan] = {}

    def register_video(
        self,
        *,
        file_path: str,
        duration_sec: float,
        video_id: str | None = None,
    ) -> VideoRef:
        resolved_video_id = video_id or _new_id("vid")
        video = VideoRef(
            id=resolved_video_id,
            file_path=file_path,
            duration_sec=float(duration_sec),
        )
        self._videos[video.id] = video
        self._clips_by_video.setdefault(video.id, [])
        return video

    def save_clips(
        self,
        *,
        video_id: str,
        clips: Sequence[ClipRange],
    ) -> list[ClipRange]:
        self._require_video(video_id)
        validated: list[ClipRange] = []
        for clip in clips:
            if clip.video_id != video_id:
                raise ValueError("All clips must belong to the target video.")
            if clip.end_sec <= clip.start_sec:
                raise ValueError("Clip end_sec must be greater than start_sec.")
            validated.append(clip)
        self._clips_by_video[video_id] = validated
        return list(validated)

    def start_analysis(
        self,
        *,
        video_id: str,
        labels: Sequence[AimLabel | str] | None = None,
        job_id: str | None = None,
    ) -> AnalysisJob:
        self._require_video(video_id)
        job = AnalysisJob(
            id=job_id or _new_id("job"),
            video_id=video_id,
            status="queued",
            requested_labels=_normalize_labels(labels or []),
        )
        self._jobs_by_video[video_id] = job
        return job

    def save_clip_diagnoses(
        self,
        *,
        video_id: str,
        clip_diagnoses: Sequence[ClipDiagnosis],
    ) -> TrainingPlan:
        self._require_video(video_id)
        known_clip_ids = {clip.id for clip in self._clips_by_video.get(video_id, [])}
        stored: dict[str, ClipDiagnosis] = {}
        for diagnosis in clip_diagnoses:
            if known_clip_ids and diagnosis.clip_id not in known_clip_ids:
                raise ValueError(f"Unknown clip_id for video {video_id}: {diagnosis.clip_id}")
            stored[diagnosis.clip_id] = diagnosis
        self._diagnoses_by_video[video_id] = stored

        training_plan = build_training_plan(
            video_id=video_id,
            clip_diagnoses=list(stored.values()),
        )
        self._training_by_video[video_id] = training_plan

        job = self._jobs_by_video.get(video_id)
        if job is not None:
            job.status = "completed"
        return training_plan

    def build_and_save_diagnoses(
        self,
        *,
        video_id: str,
        clip_labels: dict[str, Sequence[AimLabel | str]],
        confidence_by_clip: dict[str, float] | None = None,
        metrics_by_clip: dict[str, ClipAnalysisMetrics] | None = None,
    ) -> list[ClipDiagnosis]:
        diagnoses = build_clip_diagnoses(
            clip_labels=clip_labels,
            confidence_by_clip=confidence_by_clip,
            metrics_by_clip=metrics_by_clip,
        )
        self.save_clip_diagnoses(video_id=video_id, clip_diagnoses=diagnoses)
        return diagnoses

    def analyze_saved_clips(
        self,
        *,
        video_id: str,
        analyzer_config: ClipAnalyzerConfig | None = None,
    ) -> list[ClipDiagnosis]:
        video = self._require_video(video_id)
        clips = self._clips_by_video.get(video_id, [])
        if not clips:
            raise ValueError(f"No clips registered for video {video_id}.")

        job = self._jobs_by_video.get(video_id)
        if job is not None:
            job.status = "running"

        metrics_by_clip = analyze_clips(
            video_path=video.file_path,
            clips=clips,
            config=analyzer_config,
        )
        allowed_labels = job.requested_labels if job is not None and job.requested_labels else None

        clip_labels: dict[str, list[AimLabel]] = {}
        confidence_by_clip: dict[str, float] = {}
        for clip in clips:
            metrics = metrics_by_clip.get(clip.id, ClipAnalysisMetrics())
            labels, confidence = infer_labels(metrics, allowed_labels=allowed_labels)
            clip_labels[clip.id] = labels
            confidence_by_clip[clip.id] = confidence

        return self.build_and_save_diagnoses(
            video_id=video_id,
            clip_labels=clip_labels,
            confidence_by_clip=confidence_by_clip,
            metrics_by_clip=metrics_by_clip,
        )

    def get_results(self, *, video_id: str) -> dict[str, Any]:
        self._require_video(video_id)
        video = self._videos[video_id]
        clips = self._clips_by_video.get(video_id, [])
        diagnoses = self._diagnoses_by_video.get(video_id, {})
        training_plan = self._training_by_video.get(video_id)
        job = self._jobs_by_video.get(video_id)

        status = "uploaded"
        if clips:
            status = "clips_saved"
        if job is not None:
            status = job.status
        if training_plan is not None:
            status = "completed"

        return {
            "video_id": video.id,
            "status": status,
            "duration_sec": video.duration_sec,
            "clips": [_serialize_clip_result(clip, diagnoses.get(clip.id)) for clip in clips],
            "training_plan": _serialize_training_plan(training_plan),
        }

    def get_video(self, *, video_id: str) -> VideoRef:
        return self._require_video(video_id)

    def list_clips(self, *, video_id: str) -> list[ClipRange]:
        self._require_video(video_id)
        return list(self._clips_by_video.get(video_id, []))

    def _require_video(self, video_id: str) -> VideoRef:
        video = self._videos.get(video_id)
        if video is None:
            raise KeyError(f"Unknown video_id: {video_id}")
        return video


def _serialize_clip_result(clip: ClipRange, diagnosis: ClipDiagnosis | None) -> dict[str, Any]:
    base = {
        "clip_id": clip.id,
        "start_sec": round(float(clip.start_sec), 3),
        "end_sec": round(float(clip.end_sec), 3),
        "note": clip.note,
    }
    if diagnosis is None:
        base.update(
            {
                "summary": "",
                "strengths": [],
                "weaknesses": [],
                "labels": [],
                "confidence": 0.0,
                "next_focus": "",
                "metrics": {},
            }
        )
        return base

    base.update(
        {
            "summary": diagnosis.summary,
            "strengths": list(diagnosis.strengths),
            "weaknesses": list(diagnosis.weaknesses),
            "labels": [label.value for label in diagnosis.labels],
            "confidence": round(float(diagnosis.confidence), 3),
            "next_focus": diagnosis.next_focus,
            "metrics": _serialize_metrics(diagnosis.metrics),
        }
    )
    return base


def _serialize_metrics(metrics: ClipAnalysisMetrics) -> dict[str, Any]:
    return {
        "time_to_first_shot_ms": metrics.time_to_first_shot_ms,
        "aim_path_variance": metrics.aim_path_variance,
        "tracking_error_score": metrics.tracking_error_score,
        "recoil_error_score": metrics.recoil_error_score,
        "close_range_score": metrics.close_range_score,
        "ads_usage_score": metrics.ads_usage_score,
    }


def _serialize_training_plan(training_plan: TrainingPlan | None) -> dict[str, Any] | None:
    if training_plan is None:
        return None
    return {
        "priority_labels": [label.value for label in training_plan.priority_labels],
        "drills": [
            {
                "label": drill.label.value,
                "title": drill.title,
                "minutes": drill.minutes,
                "instruction": drill.instruction,
                "check_point": drill.check_point,
            }
            for drill in training_plan.drills
        ],
        "total_minutes": training_plan.total_minutes,
    }


def _normalize_labels(labels: Sequence[AimLabel | str]) -> list[AimLabel]:
    out: list[AimLabel] = []
    seen: set[AimLabel] = set()
    for raw in labels:
        label = _coerce_label(raw)
        if label is None or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _coerce_label(value: AimLabel | str) -> AimLabel | None:
    if isinstance(value, AimLabel):
        return value
    try:
        return AimLabel(str(value))
    except ValueError:
        return None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"
