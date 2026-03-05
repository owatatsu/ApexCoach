from __future__ import annotations

from collections import Counter
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Iterator

from apexcoach.action_arbiter import ActionArbiter
from apexcoach.capture_service import ScreenCaptureService, VideoCaptureService
from apexcoach.config import ApexCoachConfig
from apexcoach.event_detector import EventDetector
from apexcoach.llm_advisor import LlmAdvisor
from apexcoach.models import (
    Action,
    ArbiterResult,
    Decision,
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


class OfflinePipeline:
    def __init__(self, config: ApexCoachConfig) -> None:
        self.config = config

    def run(self) -> dict[str, int]:
        video_path = self.config.offline.input_video
        if not video_path:
            raise ValueError("Offline input_video is required.")
        _configure_opencv_threads(self.config.performance.opencv_threads)

        telemetry_reader = None
        if self.config.offline.telemetry_jsonl:
            telemetry_reader = TelemetryReader(self.config.offline.telemetry_jsonl)

        ui_parser = SimpleUiParser(telemetry=telemetry_reader)
        roi_manager = RoiManager(
            self.config.rois,
            scale_to_frame=self.config.scale_rois_to_frame,
            reference_width=self.config.roi_reference_width,
            reference_height=self.config.roi_reference_height,
        )
        event_detector = EventDetector(
            vitals_confidence_min=self.config.thresholds.vitals_confidence_min,
            min_damage_event_delta=self.config.thresholds.min_damage_event_delta,
        )
        state_aggregator = StateAggregator(
            knock_recent_seconds=self.config.thresholds.knock_recent_seconds,
            under_fire_damage_1s=self.config.thresholds.under_fire_damage_1s,
            retreat_low_total_hp_shield=self.config.thresholds.low_total_hp_shield,
            heal_total_hp_shield=self.config.thresholds.heal_total_hp_shield,
            vitals_confidence_min=self.config.thresholds.vitals_confidence_min,
            movement_score_threshold=self.config.thresholds.movement_score_threshold,
        )
        decision_engine = RuleDecisionEngine(self.config.thresholds)
        arbiter = ActionArbiter(self.config.arbiter)
        overlay = OverlayRenderer(self.config.overlay)
        llm = LlmAdvisor(self.config.llm)
        logger = SessionLogger(
            path=self.config.logging.path,
            enabled=self.config.logging.enabled,
        )

        ui_gate = RateGate(self.config.frequencies.ui_parse_fps)
        ocr_gate = RateGate(self.config.frequencies.ocr_fps)
        decision_gate = RateGate(self.config.frequencies.decision_fps)
        llm_gate = RateGate(self.config.frequencies.llm_fps)
        overlay_gate = RateGate(self.config.frequencies.overlay_fps)

        status = ParsedStatus(hp_pct=1.0, shield_pct=1.0, allies_alive=3, allies_down=0)
        tactical = ParsedTactical()
        decision = Decision(action=Action.NONE, reason="init", confidence=0.0)
        display_lines: list[str] | None = None
        llm_reason: str | None = None
        arbiter_result = ArbiterResult(
            action=Action.NONE,
            emitted=False,
            reason="init",
            source_action=Action.NONE,
            debug_notes=[],
        )

        action_counts: Counter[str] = Counter()
        writer = None
        async_writer: AsyncFrameWriter | None = None
        prefetch_stream: PrefetchFrameStream | None = None
        frames = 0

        try:
            with VideoCaptureService(
                video_path=video_path,
                target_fps=self.config.frequencies.capture_fps,
            ) as capture:
                if self.config.offline.output_video:
                    writer = _create_video_writer(
                        self.config.offline.output_video,
                        width=capture.width,
                        height=capture.height,
                        fps=max(1.0, capture.source_fps or self.config.frequencies.capture_fps),
                    )
                    if self.config.performance.parallel_io:
                        async_writer = AsyncFrameWriter(
                            writer=writer,
                            queue_size=self.config.performance.write_queue_size,
                        )

                packet_iter: Iterator = capture.iter_frames()
                if self.config.performance.parallel_io:
                    prefetch_stream = PrefetchFrameStream(
                        source=packet_iter,
                        queue_size=self.config.performance.read_prefetch_queue_size,
                    )
                    packet_iter = iter(prefetch_stream)

                for packet in packet_iter:
                    frames += 1
                    llm_reason = None
                    roi_boxes = roi_manager.resolve_boxes(packet.frame)
                    rois = roi_manager.crop(packet.frame, boxes=roi_boxes)

                    if ui_gate.ready(packet.timestamp):
                        status = ui_parser.parse_status(packet, rois)
                        tactical = ui_parser.parse_tactical(packet, rois)

                    notifications = ParsedNotifications()
                    if ocr_gate.ready(packet.timestamp):
                        notifications = ui_parser.parse_notifications(packet, rois)

                    events = event_detector.detect(
                        status=status,
                        notifications=notifications,
                        timestamp=packet.timestamp,
                    )
                    state = state_aggregator.update(
                        status=status,
                        events=events,
                        tactical=tactical,
                    )

                    if decision_gate.ready(packet.timestamp):
                        candidates = decision_engine.decide_candidates(state)
                        rule_decision = (
                            candidates[0]
                            if candidates
                            else Decision(
                                action=Action.NONE,
                                reason="No strong signal.",
                                confidence=0.5,
                            )
                        )
                        advised_decision, llm_note = llm.maybe_advise_decision(
                            state=state,
                            candidates=candidates,
                            rule_decision=rule_decision,
                            timestamp=packet.timestamp,
                            run_now=llm_gate.ready(packet.timestamp),
                        )
                        decision = advised_decision or rule_decision
                        llm_reason = llm_note
                        display_lines = _format_display_lines(
                            candidates, max_lines=self.config.overlay.max_lines
                        )
                        arbiter_result = arbiter.arbitrate(decision, packet.timestamp)
                        if state.last_action != arbiter_result.action:
                            state_aggregator.record_action(arbiter_result.action, packet.timestamp)
                            state.last_action = arbiter_result.action
                            state.last_action_time = packet.timestamp
                        action_counts[arbiter_result.action.value] += 1

                    log_llm_reason = llm_reason
                    overlay_llm_reason = _to_overlay_llm_message(llm_reason)
                    explain_reason = llm.maybe_explain(
                        state=state,
                        decision=decision,
                        arbiter=arbiter_result,
                        run_now=llm_gate.ready(packet.timestamp),
                    )
                    if explain_reason:
                        log_llm_reason = explain_reason
                        overlay_llm_reason = explain_reason

                    output_frame = packet.frame
                    if overlay_gate.ready(packet.timestamp):
                        output_frame = overlay.render(
                            frame=packet.frame,
                            action=arbiter_result.action,
                            reason=decision.reason,
                            timestamp=packet.timestamp,
                            decision_lines=display_lines,
                            llm_message=overlay_llm_reason,
                            roi_boxes=roi_boxes,
                        )

                    if async_writer is not None:
                        async_writer.write(output_frame)
                    elif writer is not None:
                        writer.write(output_frame)

                    logger.log_frame(
                        packet=packet,
                        state=state,
                        events=events,
                        decision=decision,
                        arbiter=arbiter_result,
                        llm_reason=log_llm_reason,
                    )
        finally:
            if prefetch_stream is not None:
                prefetch_stream.close()
            if async_writer is not None:
                async_writer.close()
            elif writer is not None:
                writer.release()
            overlay.close()
            logger.close()

        if frames == 0:
            raise RuntimeError(
                "No decodable frames were read from input video. "
                "This often happens with unsupported AV1 streams in the current OpenCV/FFmpeg build. "
                "Transcode the input to H.264 (yuv420p) and retry."
            )

        summary = {
            "frames": frames,
            "NONE": action_counts.get("NONE", 0),
            "HEAL": action_counts.get("HEAL", 0),
            "RETREAT": action_counts.get("RETREAT", 0),
            "TAKE_COVER": action_counts.get("TAKE_COVER", 0),
            "TAKE_HIGH_GROUND": action_counts.get("TAKE_HIGH_GROUND", 0),
            "PUSH": action_counts.get("PUSH", 0),
        }

        llm.generate_offline_review(
            session_log_path=self.config.logging.path,
            summary=summary,
        )

        return summary


class RealtimePipeline:
    def __init__(self, config: ApexCoachConfig) -> None:
        self.config = config

    def run(self) -> dict[str, int]:
        _configure_opencv_threads(self.config.performance.opencv_threads)
        telemetry_reader = None
        if self.config.offline.telemetry_jsonl:
            telemetry_reader = TelemetryReader(self.config.offline.telemetry_jsonl)

        ui_parser = SimpleUiParser(telemetry=telemetry_reader)
        roi_manager = RoiManager(
            self.config.rois,
            scale_to_frame=self.config.scale_rois_to_frame,
            reference_width=self.config.roi_reference_width,
            reference_height=self.config.roi_reference_height,
        )
        event_detector = EventDetector(
            vitals_confidence_min=self.config.thresholds.vitals_confidence_min,
            min_damage_event_delta=self.config.thresholds.min_damage_event_delta,
        )
        state_aggregator = StateAggregator(
            knock_recent_seconds=self.config.thresholds.knock_recent_seconds,
            under_fire_damage_1s=self.config.thresholds.under_fire_damage_1s,
            retreat_low_total_hp_shield=self.config.thresholds.low_total_hp_shield,
            heal_total_hp_shield=self.config.thresholds.heal_total_hp_shield,
            vitals_confidence_min=self.config.thresholds.vitals_confidence_min,
            movement_score_threshold=self.config.thresholds.movement_score_threshold,
        )
        decision_engine = RuleDecisionEngine(self.config.thresholds)
        arbiter = ActionArbiter(self.config.arbiter)
        overlay = OverlayRenderer(self.config.overlay)
        llm = LlmAdvisor(self.config.llm)
        logger = SessionLogger(
            path=self.config.logging.path,
            enabled=self.config.logging.enabled,
        )

        ui_gate = RateGate(self.config.frequencies.ui_parse_fps)
        ocr_gate = RateGate(self.config.frequencies.ocr_fps)
        decision_gate = RateGate(self.config.frequencies.decision_fps)
        llm_gate = RateGate(self.config.frequencies.llm_fps)
        overlay_gate = RateGate(self.config.frequencies.overlay_fps)

        status = ParsedStatus(hp_pct=1.0, shield_pct=1.0, allies_alive=3, allies_down=0)
        tactical = ParsedTactical()
        decision = Decision(action=Action.NONE, reason="init", confidence=0.0)
        display_lines: list[str] | None = None
        llm_reason: str | None = None
        arbiter_result = ArbiterResult(
            action=Action.NONE,
            emitted=False,
            reason="init",
            source_action=Action.NONE,
            debug_notes=[],
        )

        action_counts: Counter[str] = Counter()
        writer = None
        async_writer: AsyncFrameWriter | None = None
        prefetch_stream: PrefetchFrameStream | None = None
        frames = 0

        region = _resolve_realtime_region(self.config)
        duration_seconds = max(0.0, float(self.config.realtime.duration_seconds))

        try:
            with ScreenCaptureService(
                target_fps=self.config.frequencies.capture_fps,
                monitor_index=self.config.realtime.monitor_index,
                region=region,
            ) as capture:
                if self.config.offline.output_video:
                    writer = _create_video_writer(
                        self.config.offline.output_video,
                        width=capture.width,
                        height=capture.height,
                        fps=max(1.0, float(self.config.frequencies.capture_fps)),
                    )
                    if self.config.performance.parallel_io:
                        async_writer = AsyncFrameWriter(
                            writer=writer,
                            queue_size=self.config.performance.write_queue_size,
                        )

                # mss capture object is thread-affine on Windows.
                # Realtime frame grabbing must remain on the same thread.
                packet_iter: Iterator = capture.iter_frames()

                for packet in packet_iter:
                    if duration_seconds > 0.0 and packet.timestamp >= duration_seconds:
                        break

                    frames += 1
                    llm_reason = None
                    roi_boxes = roi_manager.resolve_boxes(packet.frame)
                    rois = roi_manager.crop(packet.frame, boxes=roi_boxes)

                    if ui_gate.ready(packet.timestamp):
                        status = ui_parser.parse_status(packet, rois)
                        tactical = ui_parser.parse_tactical(packet, rois)

                    notifications = ParsedNotifications()
                    if ocr_gate.ready(packet.timestamp):
                        notifications = ui_parser.parse_notifications(packet, rois)

                    events = event_detector.detect(
                        status=status,
                        notifications=notifications,
                        timestamp=packet.timestamp,
                    )
                    state = state_aggregator.update(
                        status=status,
                        events=events,
                        tactical=tactical,
                    )

                    if decision_gate.ready(packet.timestamp):
                        candidates = decision_engine.decide_candidates(state)
                        rule_decision = (
                            candidates[0]
                            if candidates
                            else Decision(
                                action=Action.NONE,
                                reason="No strong signal.",
                                confidence=0.5,
                            )
                        )
                        advised_decision, llm_note = llm.maybe_advise_decision(
                            state=state,
                            candidates=candidates,
                            rule_decision=rule_decision,
                            timestamp=packet.timestamp,
                            run_now=llm_gate.ready(packet.timestamp),
                        )
                        decision = advised_decision or rule_decision
                        llm_reason = llm_note
                        display_lines = _format_display_lines(
                            candidates, max_lines=self.config.overlay.max_lines
                        )
                        arbiter_result = arbiter.arbitrate(decision, packet.timestamp)
                        if state.last_action != arbiter_result.action:
                            state_aggregator.record_action(arbiter_result.action, packet.timestamp)
                            state.last_action = arbiter_result.action
                            state.last_action_time = packet.timestamp
                        action_counts[arbiter_result.action.value] += 1

                    log_llm_reason = llm_reason
                    overlay_llm_reason = _to_overlay_llm_message(llm_reason)
                    explain_reason = llm.maybe_explain(
                        state=state,
                        decision=decision,
                        arbiter=arbiter_result,
                        run_now=llm_gate.ready(packet.timestamp),
                    )
                    if explain_reason:
                        log_llm_reason = explain_reason
                        overlay_llm_reason = explain_reason

                    output_frame = packet.frame
                    if overlay_gate.ready(packet.timestamp):
                        output_frame = overlay.render(
                            frame=packet.frame,
                            action=arbiter_result.action,
                            reason=decision.reason,
                            timestamp=packet.timestamp,
                            decision_lines=display_lines,
                            llm_message=overlay_llm_reason,
                            roi_boxes=roi_boxes,
                        )

                    if async_writer is not None:
                        async_writer.write(output_frame)
                    elif writer is not None:
                        writer.write(output_frame)

                    logger.log_frame(
                        packet=packet,
                        state=state,
                        events=events,
                        decision=decision,
                        arbiter=arbiter_result,
                        llm_reason=log_llm_reason,
                    )
        except KeyboardInterrupt:
            pass
        finally:
            if prefetch_stream is not None:
                prefetch_stream.close()
            if async_writer is not None:
                async_writer.close()
            elif writer is not None:
                writer.release()
            overlay.close()
            logger.close()

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
        out.append(f"{d.action.value} | {d.reason}")
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
