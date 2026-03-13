from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Any, Iterator

from apexcoach.action_arbiter import ActionArbiter
from apexcoach.capture_service import ScreenCaptureService, VideoCaptureService
from apexcoach.config import ApexCoachConfig, format_run_timestamp
from apexcoach.display_text import format_instruction_line, localize_reason
from apexcoach.event_detector import EventDetector
from apexcoach.llm_advisor import LlmAdvisor
from apexcoach.models import (
    Action,
    ArbiterResult,
    Decision,
    FramePacket,
    GameState,
    ParsedNotifications,
    ParsedStatus,
    ParsedTactical,
)
from apexcoach.overlay_renderer import OverlayRenderer
from apexcoach.roi_manager import RoiManager
from apexcoach.rule_decision_engine import RuleDecisionEngine
from apexcoach.session_logger import SessionLogger
from apexcoach.state_aggregator import StateAggregator
from apexcoach.ui_parser import SimpleUiParser, TelemetryReader

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


class RateGate:
    def __init__(self, fps: int) -> None:
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self.next_time: float | None = None

    def ready(self, timestamp: float) -> bool:
        if self.interval <= 0.0:
            return True
        if self.next_time is None:
            self.next_time = timestamp
        if timestamp + 1e-9 < self.next_time:
            return False
        while self.next_time <= timestamp + 1e-9:
            self.next_time += self.interval
        return True


@dataclass(slots=True)
class _PipelineRuntime:
    status: ParsedStatus = field(
        default_factory=lambda: ParsedStatus(
            hp_pct=1.0,
            shield_pct=1.0,
            allies_alive=3,
            allies_down=0,
        )
    )
    tactical: ParsedTactical = field(default_factory=ParsedTactical)
    decision: Decision = field(
        default_factory=lambda: Decision(
            action=Action.NONE,
            reason="init",
            confidence=0.0,
        )
    )
    display_lines: list[str] | None = None
    arbiter_result: ArbiterResult = field(
        default_factory=lambda: ArbiterResult(
            action=Action.NONE,
            emitted=False,
            reason="init",
            source_action=Action.NONE,
            debug_notes=[],
        )
    )
    action_counts: Counter[str] = field(default_factory=Counter)


class _PipelineSession:
    def __init__(self, config: ApexCoachConfig) -> None:
        self.config = config
        telemetry_reader = None
        if self.config.offline.telemetry_jsonl:
            telemetry_reader = TelemetryReader(self.config.offline.telemetry_jsonl)

        self.ui_parser = SimpleUiParser(telemetry=telemetry_reader)
        self.roi_manager = RoiManager(
            self.config.rois,
            scale_to_frame=self.config.scale_rois_to_frame,
            reference_width=self.config.roi_reference_width,
            reference_height=self.config.roi_reference_height,
        )
        self.event_detector = EventDetector(
            vitals_confidence_min=self.config.thresholds.vitals_confidence_min,
            min_damage_event_delta=self.config.thresholds.min_damage_event_delta,
        )
        self.state_aggregator = StateAggregator(
            knock_recent_seconds=self.config.thresholds.knock_recent_seconds,
            under_fire_damage_1s=self.config.thresholds.under_fire_damage_1s,
            retreat_low_total_hp_shield=self.config.thresholds.low_total_hp_shield,
            heal_total_hp_shield=self.config.thresholds.heal_total_hp_shield,
            vitals_confidence_min=self.config.thresholds.vitals_confidence_min,
            movement_score_threshold=self.config.thresholds.movement_score_threshold,
        )
        self.decision_engine = RuleDecisionEngine(self.config.thresholds)
        self.arbiter = ActionArbiter(self.config.arbiter)
        self.overlay = OverlayRenderer(self.config.overlay)
        self.llm = LlmAdvisor(self.config.llm)
        self.logger = SessionLogger(
            path=self.config.logging.path,
            enabled=self.config.logging.enabled,
        )

        self.ui_gate = RateGate(self.config.frequencies.ui_parse_fps)
        self.ocr_gate = RateGate(self.config.frequencies.ocr_fps)
        self.decision_gate = RateGate(self.config.frequencies.decision_fps)
        self.llm_gate = RateGate(self.config.frequencies.llm_fps)
        self.overlay_gate = RateGate(self.config.frequencies.overlay_fps)

        self.runtime = _PipelineRuntime()
        self.frames = 0
        self.writer: Any | None = None
        self.async_writer: AsyncFrameWriter | None = None

    def configure_output(self, width: int, height: int, fps: float) -> None:
        if not self.config.offline.output_video:
            return

        self.writer = _create_video_writer(
            self.config.offline.output_video,
            width=width,
            height=height,
            fps=fps,
        )
        if self.config.performance.parallel_io:
            self.async_writer = AsyncFrameWriter(
                writer=self.writer,
                queue_size=self.config.performance.write_queue_size,
            )

    def process_packet(self, packet: FramePacket) -> Any:
        self.frames += 1
        runtime = self.runtime
        llm_reason: str | None = None

        roi_boxes = self.roi_manager.resolve_boxes(packet.frame)
        rois = self.roi_manager.crop(packet.frame, boxes=roi_boxes)

        if self.ui_gate.ready(packet.timestamp):
            runtime.status = self.ui_parser.parse_status(packet, rois)
            runtime.tactical = self.ui_parser.parse_tactical(packet, rois)

        notifications = ParsedNotifications()
        if self.ocr_gate.ready(packet.timestamp):
            notifications = self.ui_parser.parse_notifications(packet, rois)

        events = self.event_detector.detect(
            status=runtime.status,
            notifications=notifications,
            timestamp=packet.timestamp,
        )
        state = self.state_aggregator.update(
            status=runtime.status,
            events=events,
            tactical=runtime.tactical,
        )

        if self.decision_gate.ready(packet.timestamp):
            llm_reason = self._update_decision(state, packet.timestamp)

        log_llm_reason = llm_reason
        overlay_llm_reason = _to_overlay_llm_message(llm_reason)
        explain_reason = self.llm.maybe_explain(
            state=state,
            decision=runtime.decision,
            arbiter=runtime.arbiter_result,
            run_now=self.llm_gate.ready(packet.timestamp),
        )
        if explain_reason:
            log_llm_reason = explain_reason
            overlay_llm_reason = explain_reason

        output_frame = packet.frame
        if self.overlay_gate.ready(packet.timestamp):
            output_frame = self.overlay.render(
                frame=packet.frame,
                action=runtime.arbiter_result.action,
                reason=localize_reason(runtime.decision.reason),
                timestamp=packet.timestamp,
                decision_lines=runtime.display_lines,
                llm_message=overlay_llm_reason,
                roi_boxes=roi_boxes,
            )

        self.logger.log_frame(
            packet=packet,
            state=state,
            events=events,
            decision=runtime.decision,
            arbiter=runtime.arbiter_result,
            llm_reason=log_llm_reason,
        )
        return output_frame

    def write_output(self, frame: Any) -> None:
        if self.async_writer is not None:
            self.async_writer.write(frame)
            return
        if self.writer is not None:
            self.writer.write(frame)

    def summary(self) -> dict[str, int]:
        return _build_summary(self.frames, self.runtime.action_counts)

    def generate_offline_review(self) -> str | None:
        return self.llm.generate_offline_review(
            session_log_path=self.config.logging.path,
            summary=self.summary(),
        )

    def close(self) -> None:
        if self.async_writer is not None:
            self.async_writer.close()
            self.async_writer = None
            self.writer = None
        elif self.writer is not None:
            self.writer.release()
            self.writer = None
        self.overlay.close()
        self.logger.close()

    def _update_decision(self, state: GameState, timestamp: float) -> str | None:
        runtime = self.runtime
        candidates = self.decision_engine.decide_candidates(state)
        rule_decision = _select_rule_decision(candidates)
        advised_decision, llm_reason = self.llm.maybe_advise_decision(
            state=state,
            candidates=candidates,
            rule_decision=rule_decision,
            timestamp=timestamp,
            run_now=self.llm_gate.ready(timestamp),
        )
        runtime.decision = advised_decision or rule_decision
        runtime.display_lines = _format_display_lines(
            candidates,
            max_lines=self.config.overlay.max_lines,
        )
        runtime.arbiter_result = self.arbiter.arbitrate(runtime.decision, timestamp)
        _record_action_transition(
            state_aggregator=self.state_aggregator,
            state=state,
            action=runtime.arbiter_result.action,
            timestamp=timestamp,
        )
        runtime.action_counts[runtime.arbiter_result.action.value] += 1
        return llm_reason


class OfflinePipeline:
    def __init__(self, config: ApexCoachConfig) -> None:
        self.config = config

    def run(self) -> dict[str, int]:
        video_path = self.config.offline.input_video
        if not video_path:
            raise ValueError("Offline input_video is required.")
        _configure_opencv_threads(self.config.performance.opencv_threads)
        _resolve_run_artifact_paths(self.config)

        session = _PipelineSession(self.config)
        prefetch_stream: PrefetchFrameStream | None = None

        try:
            with VideoCaptureService(
                video_path=video_path,
                target_fps=self.config.frequencies.capture_fps,
            ) as capture:
                session.configure_output(
                    width=capture.width,
                    height=capture.height,
                    fps=max(
                        1.0,
                        capture.source_fps or self.config.frequencies.capture_fps,
                    ),
                )

                packet_iter: Iterator = capture.iter_frames()
                if self.config.performance.parallel_io:
                    prefetch_stream = PrefetchFrameStream(
                        source=packet_iter,
                        queue_size=self.config.performance.read_prefetch_queue_size,
                    )
                    packet_iter = iter(prefetch_stream)

                for packet in packet_iter:
                    session.write_output(session.process_packet(packet))
        finally:
            if prefetch_stream is not None:
                prefetch_stream.close()
            session.close()

        if session.frames == 0:
            raise RuntimeError(
                "No decodable frames were read from input video. "
                "This often happens with unsupported AV1 streams in the current OpenCV/FFmpeg build. "
                "Transcode the input to H.264 (yuv420p) and retry."
            )

        summary = session.summary()
        session.generate_offline_review()

        return summary


class RealtimePipeline:
    def __init__(self, config: ApexCoachConfig) -> None:
        self.config = config

    def run(self) -> dict[str, int]:
        _configure_opencv_threads(self.config.performance.opencv_threads)
        _resolve_run_artifact_paths(self.config)
        session = _PipelineSession(self.config)

        region = _resolve_realtime_region(self.config)
        duration_seconds = max(0.0, float(self.config.realtime.duration_seconds))

        try:
            with ScreenCaptureService(
                target_fps=self.config.frequencies.capture_fps,
                monitor_index=self.config.realtime.monitor_index,
                region=region,
            ) as capture:
                session.configure_output(
                    width=capture.width,
                    height=capture.height,
                    fps=max(1.0, float(self.config.frequencies.capture_fps)),
                )

                # mss capture object is thread-affine on Windows.
                # Realtime frame grabbing must remain on the same thread.
                packet_iter: Iterator = capture.iter_frames()

                for packet in packet_iter:
                    if duration_seconds > 0.0 and packet.timestamp >= duration_seconds:
                        break

                    session.write_output(session.process_packet(packet))
        except KeyboardInterrupt:
            pass
        finally:
            session.close()

        return session.summary()


def _select_rule_decision(candidates: list[Decision]) -> Decision:
    if candidates:
        return candidates[0]
    return Decision(
        action=Action.NONE,
        reason="No strong signal.",
        confidence=0.5,
    )


def _record_action_transition(
    state_aggregator: StateAggregator,
    state: GameState,
    action: Action,
    timestamp: float,
) -> None:
    if state.last_action == action:
        return
    state_aggregator.record_action(action, timestamp)
    state.last_action = action
    state.last_action_time = timestamp


def _build_summary(frames: int, action_counts: Counter[str]) -> dict[str, int]:
    return {
        "frames": frames,
        "NONE": action_counts.get("NONE", 0),
        "HEAL": action_counts.get("HEAL", 0),
        "RETREAT": action_counts.get("RETREAT", 0),
        "TAKE_COVER": action_counts.get("TAKE_COVER", 0),
        "TAKE_HIGH_GROUND": action_counts.get("TAKE_HIGH_GROUND", 0),
        "PUSH": action_counts.get("PUSH", 0),
    }


def _create_video_writer(path: str, width: int, height: int, fps: float):
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video output.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {output_path}")
    return writer


def _resolve_realtime_region(
    config: ApexCoachConfig,
) -> tuple[int, int, int, int] | None:
    w = int(config.realtime.region_w)
    h = int(config.realtime.region_h)
    if w <= 0 or h <= 0:
        return None
    return (
        int(config.realtime.region_x),
        int(config.realtime.region_y),
        w,
        h,
    )


def _format_display_lines(candidates: list[Decision], max_lines: int) -> list[str]:
    limit = max(1, int(max_lines))
    out: list[str] = []
    for d in candidates[:limit]:
        out.append(format_instruction_line(d.action, d.reason))
    return out


def _to_overlay_llm_message(raw_reason: str | None) -> str | None:
    text = (raw_reason or "").strip()
    if not text:
        return None

    lowered = text.lower()
    internal_prefixes = (
        "llm_skip:",
        "llm_none",
        "provider_disabled",
        "network_error",
        "timeout",
        "http_",
    )
    if lowered.startswith(internal_prefixes):
        return None
    return text


def _configure_opencv_threads(thread_count: int) -> None:
    if cv2 is None:
        return
    if thread_count <= 0:
        return
    cv2.setNumThreads(int(thread_count))


def _resolve_run_artifact_paths(
    config: ApexCoachConfig,
    now: datetime | None = None,
) -> None:
    timestamp = format_run_timestamp(now)
    config.logging.path = _expand_run_path(config.logging.path, timestamp)
    config.llm.offline_review_output = _expand_run_path(
        config.llm.offline_review_output,
        timestamp,
    )


def _expand_run_path(path: str, timestamp: str) -> str:
    value = str(path or "").strip()
    if not value:
        return value
    if "{timestamp}" in value:
        return value.replace("{timestamp}", timestamp)
    return value


_SENTINEL = object()


class PrefetchFrameStream:
    def __init__(self, source: Iterator, queue_size: int = 64) -> None:
        self._source = source
        self._queue: Queue = Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = Event()
        self._error: BaseException | None = None
        self._thread = Thread(target=self._run, name="apexcoach-prefetch", daemon=True)
        self._thread.start()

    def __iter__(self) -> Iterator:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item

        if self._error is not None:
            raise RuntimeError("Prefetch frame reader failed.") from self._error

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            for item in self._source:
                if self._stop_event.is_set():
                    break
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except Full:
                        continue
        except BaseException as exc:
            self._error = exc
        finally:
            while True:
                try:
                    self._queue.put(_SENTINEL, timeout=0.1)
                    break
                except Full:
                    if self._stop_event.is_set():
                        continue


class AsyncFrameWriter:
    def __init__(self, writer, queue_size: int = 64) -> None:
        self._writer = writer
        self._queue: Queue = Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = Event()
        self._error: BaseException | None = None
        self._thread = Thread(target=self._run, name="apexcoach-writer", daemon=True)
        self._thread.start()

    def write(self, frame) -> None:
        if self._error is not None:
            raise RuntimeError("Async frame writer failed.") from self._error

        while not self._stop_event.is_set():
            try:
                self._queue.put(frame, timeout=0.1)
                return
            except Full:
                if self._error is not None:
                    raise RuntimeError("Async frame writer failed.") from self._error
                continue

    def close(self) -> None:
        self._stop_event.set()
        while True:
            try:
                self._queue.put(_SENTINEL, timeout=0.1)
                break
            except Full:
                continue
        self._thread.join(timeout=10.0)
        self._writer.release()
        if self._error is not None:
            raise RuntimeError("Async frame writer failed.") from self._error

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                self._writer.write(item)
        except BaseException as exc:
            self._error = exc
