from __future__ import annotations

from collections import Counter
from pathlib import Path

from apexcoach.action_arbiter import ActionArbiter
from apexcoach.capture_service import VideoCaptureService
from apexcoach.config import ApexCoachConfig
from apexcoach.event_detector import EventDetector
from apexcoach.llm_advisor import LlmAdvisor
from apexcoach.models import Action, ArbiterResult, Decision, ParsedNotifications, ParsedStatus
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

        telemetry_reader = None
        if self.config.offline.telemetry_jsonl:
            telemetry_reader = TelemetryReader(self.config.offline.telemetry_jsonl)

        ui_parser = SimpleUiParser(telemetry=telemetry_reader)
        roi_manager = RoiManager(self.config.rois)
        event_detector = EventDetector()
        state_aggregator = StateAggregator(
            knock_recent_seconds=self.config.thresholds.knock_recent_seconds,
            under_fire_damage_1s=self.config.thresholds.under_fire_damage_1s,
        )
        decision_engine = RuleDecisionEngine(self.config.thresholds)
        arbiter = ActionArbiter(self.config.arbiter)
        overlay = OverlayRenderer(self.config.overlay)
        llm = LlmAdvisor(enabled=self.config.llm.enabled)
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
        decision = Decision(action=Action.NONE, reason="init", confidence=0.0)
        arbiter_result = ArbiterResult(
            action=Action.NONE,
            emitted=False,
            reason="init",
            source_action=Action.NONE,
            debug_notes=[],
        )

        action_counts: Counter[str] = Counter()
        writer = None
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

                for packet in capture.iter_frames():
                    frames += 1
                    rois = roi_manager.crop(packet.frame)

                    if ui_gate.ready(packet.timestamp):
                        status = ui_parser.parse_status(packet, rois)

                    notifications = ParsedNotifications()
                    if ocr_gate.ready(packet.timestamp):
                        notifications = ui_parser.parse_notifications(packet, rois)

                    events = event_detector.detect(
                        status=status,
                        notifications=notifications,
                        timestamp=packet.timestamp,
                    )
                    state = state_aggregator.update(status=status, events=events)

                    if decision_gate.ready(packet.timestamp):
                        decision = decision_engine.decide(state)
                        arbiter_result = arbiter.arbitrate(decision, packet.timestamp)
                        if state.last_action != arbiter_result.action:
                            state_aggregator.record_action(arbiter_result.action, packet.timestamp)
                            state.last_action = arbiter_result.action
                            state.last_action_time = packet.timestamp
                        action_counts[arbiter_result.action.value] += 1

                    llm_reason = llm.maybe_explain(
                        state=state,
                        decision=decision,
                        arbiter=arbiter_result,
                        run_now=llm_gate.ready(packet.timestamp),
                    )

                    output_frame = packet.frame
                    if overlay_gate.ready(packet.timestamp):
                        output_frame = overlay.render(
                            frame=packet.frame,
                            action=arbiter_result.action,
                            reason=decision.reason,
                        )

                    if writer is not None:
                        writer.write(output_frame)

                    logger.log_frame(
                        packet=packet,
                        state=state,
                        events=events,
                        decision=decision,
                        arbiter=arbiter_result,
                        llm_reason=llm_reason,
                    )
        finally:
            if writer is not None:
                writer.release()
            overlay.close()
            logger.close()

        if frames == 0:
            raise RuntimeError(
                "No decodable frames were read from input video. "
                "This often happens with unsupported AV1 streams in the current OpenCV/FFmpeg build. "
                "Transcode the input to H.264 (yuv420p) and retry."
            )

        return {
            "frames": frames,
            "NONE": action_counts.get("NONE", 0),
            "HEAL": action_counts.get("HEAL", 0),
            "RETREAT": action_counts.get("RETREAT", 0),
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
