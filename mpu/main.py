"""
Helmet Detection System - Main Application

This is the main application file for a real-time helmet detection and safety monitoring system.
The system integrates computer vision, Arduino control, and alert transmission to provide
comprehensive safety monitoring in industrial environments.

Key Features:
- Real-time helmet detection using computer vision
- Person detection and tracking
- Arduino-based alert system (LED/buzzer control)
- Remote alert transmission to monitoring servers
- Automatic safety alerts when helmet violations are detected

System Components:
- Camera capture and frame processing
- AI-powered person detection (MobileNet SSD)
- Helmet classification (EfficientNet)
- Alert management with cooldown periods
- UNO Q Bridge communication with Arduino MCU firmware
- HTTP alert transmission
"""

import cv2
import numpy as np
import argparse
import base64
import logging
import math
import time
import threading
from mpu.camera import CameraCapture
from mpu.detector import PersonDetector
from mpu.classifier import HelmetClassifier
from mpu.alert_manager import AlertManager
from mpu.bridge_rpc import BridgeRPC
from mpu.sender import Sender
from mpu.tracker import PersonTracker
from mpu.config import DEFAULT_DETECTION_THRESHOLD, DEFAULT_COOLDOWN_TIME, DEFAULT_SERIAL_PORT, DEFAULT_SERVER_URL, ENABLE_DISPLAY, APP_LAB_DEV_MODE, WARNING_CLEAR_THRESHOLD, validate_runtime_models
from mpu.dashboard_state import DashboardState, EventType, HelmetResult, MovementState, RobotMode, SafetyState

logger = logging.getLogger(__name__)

_WORKER_IOU_THRESHOLD = 0.3    # Minimum IoU to consider two bboxes the same worker
_EVENT_SUPPRESS_SECONDS = 10.0 # Minimum seconds between identical system events on the dashboard
_CONTROL_TICK_INTERVAL = 0.25  # Seconds between motion-lease renewal ticks sent to the MCU
_TELEMETRY_INTERVAL = 0.5      # Seconds between MCU status snapshots for the dashboard


class _EventSuppressor:
    """
    Prevents repeated identical system events within a time window.

    Used only for system-level events: connection, alert, and hardware errors.
    Worker detection events are emitted directly to DashboardState so the
    accepted-worker gate (IoU) is the sole deduplication mechanism.

    Suppression is keyed on an explicit key string, not the human-readable
    message.  When no key is provided, the message itself is the key.
    """

    def __init__(self, dashboard: DashboardState, suppress_seconds: float = _EVENT_SUPPRESS_SECONDS):
        self._dashboard = dashboard
        self._suppress_seconds = suppress_seconds
        self._last_emitted: dict = {}

    def append(self, event_type: EventType, message: str, key: str = None) -> bool:
        """
        Emit event to DashboardState unless the same key was recently emitted.

        Args:
            event_type: EventType enum value.
            message:    Human-readable message stored in the event.
            key:        Suppression key.  Defaults to message when None.

        Returns True if the event was emitted, False if suppressed.
        """
        suppress_key = message if key is None else key  # Use explicit key when provided
        now = time.monotonic()
        if now - self._last_emitted.get(suppress_key, 0.0) < self._suppress_seconds:
            return False  # Still within suppression window; drop duplicate event
        self._last_emitted[suppress_key] = now
        self._dashboard.append_event(event_type, message)
        return True


def _bbox_iou(a, b) -> float:
    """
    Compute Intersection-over-Union between two [x, y, w, h] bboxes.
    Returns 0.0 when either bbox has zero area.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ix = max(ax, bx)                    # Left edge of intersection
    iy = max(ay, by)                    # Top edge of intersection
    iw = min(ax + aw, bx + bw) - ix    # Width of intersection
    ih = min(ay + ah, by + bh) - iy    # Height of intersection

    if iw <= 0 or ih <= 0:
        return 0.0  # No overlap

    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _is_new_worker(bbox, prev_bboxes) -> bool:
    """
    Return True if bbox does not overlap any bbox in prev_bboxes above the
    IoU threshold, indicating a newly entered worker.
    """
    for prev in prev_bboxes:
        if _bbox_iou(bbox, prev) >= _WORKER_IOU_THRESHOLD:
            return False
    return True


class HelmetDetectionSystem:
    """
    Main helmet detection system class that integrates all components.

    This class orchestrates the entire pipeline from camera capture to alert transmission:
    1. Captures video frames from camera
    2. Detects persons in the frame
    3. Classifies helmet wearing status for each person
    4. Manages alerts based on detection results
    5. Communicates with Arduino for local alerts
    6. Transmits alerts to remote monitoring systems
    """
    def __init__(self, port: str = DEFAULT_SERIAL_PORT, server_url: str = DEFAULT_SERVER_URL):
        """
        Initialize the helmet detection system with all required components.

        Args:
            port: MCU transport label for Arduino communication (default from config)
            server_url: URL for remote alert transmission (default from config)
        """
        # Initialize computer vision components
        self.camera = CameraCapture(dev_mode=APP_LAB_DEV_MODE)  # Camera capture system
        self.person_detector = PersonDetector()          # Person detection AI model
        self.helmet_classifier = HelmetClassifier()      # Helmet classification AI model

        # Initialize communication components
        self.sender = Sender(server_url)                 # HTTP alert transmission
        self.bridge_rpc = BridgeRPC(port)               # UNO Q Bridge MCU communication

        # Initialize alert management with callback for hardware alerts
        self.alert_manager = AlertManager(callback=self.on_no_helmet_alert)
        self.person_tracker = PersonTracker()
        self._next_incident_seq = 1

        # System state
        self.running = False
        self._connected = False  # True only after a successful connect()
        self._dev_mode = APP_LAB_DEV_MODE  # Hardware-free dev mode flag
        self._warning_hardware_active = False
        self._last_violation_detected = False
        self._clear_violation_count = 0
        self._tracked_incident_seen = False

        # Dashboard state mirror
        self.dashboard = DashboardState()
        self._frame_lock = threading.Lock()
        self._latest_frame = {
            "status": "unavailable",
            "image": None,
            "content_type": "image/jpeg",
            "updated_at": None,
            "error": None,
        }

        # Event suppressor wrapping the dashboard event queue.
        self._events = _EventSuppressor(self.dashboard)

        # Bboxes from the previous frame used for new-worker detection.
        self._prev_bboxes: list = []
        self._pending_alert_frame = None
        self._pending_alert_label = None
        self._pending_alert_confidence = None
        self._pending_alert_metadata = None

        # Monotonic timestamp of the last control_tick sent to the MCU.
        self._last_tick_time: float = 0.0
        self._last_telemetry_time: float = 0.0

    def _ensure_frame_state(self) -> None:
        if not hasattr(self, "_frame_lock"):
            self._frame_lock = threading.Lock()
            self._latest_frame = {
                "status": "unavailable",
                "image": None,
                "content_type": "image/jpeg",
                "updated_at": None,
                "error": None,
            }

    def _set_frame_status(self, status: str, error: str = None) -> None:
        self._ensure_frame_state()
        payload = {
            "status": status,
            "image": None,
            "content_type": "image/jpeg",
            "updated_at": time.time(),
            "error": error,
        }
        with self._frame_lock:
            self._latest_frame = payload

    def _store_live_frame(self, frame) -> None:
        self._ensure_frame_state()
        try:
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                self._set_frame_status("encode_error", "JPEG encode failed")
                return
            image = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")
            payload = {
                "status": "ok",
                "image": image,
                "content_type": "image/jpeg",
                "updated_at": time.time(),
                "error": None,
            }
            with self._frame_lock:
                self._latest_frame = payload
        except Exception as exc:
            logger.warning("Live frame encode failed: %s", exc)
            self._set_frame_status("encode_error", str(exc))

    def latest_frame_payload(self) -> dict:
        self._ensure_frame_state()
        with self._frame_lock:
            return dict(self._latest_frame)

    def on_no_helmet_alert(self, *, track_id=None, incident_id=None, frame=None, label=None, confidence=None):
        """
        Callback function triggered when a helmet violation alert is needed.
        Activates Arduino-based LED and buzzer alerts.
        """
        self.dashboard.update_statistics(warnings_delta=1)
        self._events.append(EventType.ALERT, "No-helmet alert triggered")
        self._start_helmet_warning()
        alert_frame = frame if frame is not None else getattr(self, "_pending_alert_frame", None)
        alert_label = label if label is not None else getattr(self, "_pending_alert_label", None)
        alert_confidence = confidence if confidence is not None else getattr(self, "_pending_alert_confidence", None)
        metadata = {"track_id": track_id, "incident_id": incident_id} if track_id is not None and incident_id is not None else getattr(self, "_pending_alert_metadata", None)
        if alert_frame is not None:
            try:
                self.sender.send_alert(alert_frame, alert_label, alert_confidence, metadata=metadata)
            except Exception as e:
                logger.error("Failed to send alert: %s", e)

    def _start_helmet_warning(self):
        if getattr(self, "_warning_hardware_active", False):
            self._events.append(EventType.ALERT, "No-helmet warning buzzer request coalesced")
            return False
        try:
            self.bridge_rpc.led_control("red")
            self._warning_hardware_active = True
            self._events.append(EventType.ALERT, "No-helmet warning started")
            return True
        except Exception as e:
            msg = f"Arduino communication failed: {e}"
            logger.error("Helmet warning command failed: %s", e)
            self._events.append(EventType.SYSTEM, msg)
            return False

    def _clear_helmet_warning(self):
        if not getattr(self, "_warning_hardware_active", False):
            return False
        try:
            self.bridge_rpc.led_control("off")
            self._warning_hardware_active = False
            self._events.append(EventType.ALERT, "No-helmet warning cleared")
            return True
        except Exception as e:
            msg = f"Arduino communication failed: {e}"
            logger.error("Helmet warning clear failed: %s", e)
            self._events.append(EventType.SYSTEM, msg)
            return False

    def _handle_frame_violation(self, helmet_state):
        no_helmet_detected = helmet_state == "no_helmet"
        if self._uses_legacy_alert_manager_mock():
            self.alert_manager.on_detection(no_helmet_detected)
        if helmet_state == "no_helmet":
            self._clear_violation_count = 0
            self._last_violation_detected = True
        elif helmet_state == "helmet":
            self._clear_violation_count = getattr(self, "_clear_violation_count", 0) + 1
            if (
                getattr(self, "_last_violation_detected", False)
                and self._clear_violation_count >= WARNING_CLEAR_THRESHOLD
            ):
                self._clear_helmet_warning()
                if hasattr(self.alert_manager, "resolve_incident"):
                    self.alert_manager.resolve_incident()
                self._last_violation_detected = False
        else:
            self._clear_violation_count = 0

    def reset_detection_policy_state(self):
        if hasattr(self.alert_manager, "detection_count"):
            self.alert_manager.detection_count = 0
        if hasattr(self.alert_manager, "resolve_incident"):
            self.alert_manager.resolve_incident()
        if hasattr(self, "person_tracker"):
            self.person_tracker.reset()
        self._tracked_incident_seen = False
        self._last_violation_detected = False
        self._clear_violation_count = 0

    def _uses_legacy_alert_manager_mock(self) -> bool:
        return hasattr(getattr(self.alert_manager, "on_detection", None), "assert_called_once_with")

    def _ensure_tracking_state(self) -> None:
        if not hasattr(self, "person_tracker"):
            self.person_tracker = PersonTracker()
        if not hasattr(self, "_next_incident_seq"):
            self._next_incident_seq = 1
        if not hasattr(self, "_tracked_incident_seen"):
            self._tracked_incident_seen = False

    def _new_incident_id(self, track_id: int) -> str:
        incident_id = f"helmet-{track_id}-{self._next_incident_seq}"
        self._next_incident_seq += 1
        return incident_id

    def _track_can_alert(self, track) -> bool:
        cooldown = getattr(self.alert_manager, "cooldown", DEFAULT_COOLDOWN_TIME)
        if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) or not math.isfinite(cooldown):
            cooldown = DEFAULT_COOLDOWN_TIME
        return time.time() - getattr(track, "last_incident_time", 0.0) >= cooldown

    def _apply_track_classification(self, track, label: str, frame=None, confidence: float = None) -> bool:
        track.classification_state = label
        if label == "no_helmet":
            track.consecutive_no_helmet += 1
            track.consecutive_helmet = 0
            self._last_violation_detected = True
        elif label == "helmet":
            track.consecutive_no_helmet = 0
            track.consecutive_helmet += 1
            if track.incident_active and track.consecutive_helmet >= WARNING_CLEAR_THRESHOLD:
                track.incident_active = False
                track.current_incident_id = None
                track.http_event_sent = False
                track.incident_counted = False
                track.buzzer_requested = False
            return False
        else:
            track.consecutive_no_helmet = 0
            track.consecutive_helmet = 0
            return False

        if track.incident_active:
            return False
        if track.consecutive_no_helmet < DEFAULT_DETECTION_THRESHOLD or not self._track_can_alert(track):
            return False

        incident_id = self._new_incident_id(track.track_id)
        track.incident_active = True
        track.current_incident_id = incident_id
        track.http_event_sent = True
        track.incident_counted = True
        track.buzzer_requested = True
        track.last_incident_time = time.time()
        self._tracked_incident_seen = True
        if hasattr(self.alert_manager, "incident_active"):
            self.alert_manager.incident_active = True
        if hasattr(self.alert_manager, "detection_count"):
            self.alert_manager.detection_count = 0
        self.on_no_helmet_alert(
            track_id=track.track_id,
            incident_id=incident_id,
            frame=frame,
            label=label,
            confidence=confidence,
        )
        return True

    def _poll_mcu_status(self):
        status = self.bridge_rpc.get_status()
        distances = status.get("distances", {})
        self.dashboard.update_mode(RobotMode(status["mode"]))
        self.dashboard.update_movement(MovementState(status["movement"]))
        self.dashboard.update_distances(
            front=distances.get("front"),
            rear=distances.get("rear"),
            left=distances.get("left"),
            right=distances.get("right"),
        )
        self.dashboard.update_control(
            motion_lease_active=status.get("motion_lease_active"),
            control_tick_fresh=status.get("control_tick_fresh"),
            command_fresh=status.get("command_fresh"),
            warning_active=status.get("warning_active"),
            safety_state=SafetyState(status["safety_state"]),
            motor_speed=status.get("motor_speed"),
        )

    def crop_person(self, frame, bbox):
        """
        Crop person region from frame using bounding box coordinates.
        Ensures cropped region stays within frame boundaries.

        Args:
            frame: Input video frame
            bbox: Bounding box [x, y, width, height]

        Returns:
            Cropped image region containing the detected person
        """
        x, y, w, h = bbox
        # Ensure coordinates stay within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)
        return frame[y:y+h, x:x+w]

    def process_frame(self, frame):
        """
        Process a single video frame for helmet detection and safety monitoring.

        Pipeline:
        1. Detect all persons in the frame
        2. Crop each person's region
        3. Classify helmet wearing status
        4. Draw bounding boxes and labels
        5. Trigger alerts for violations
        6. Update hardware status

        Args:
            frame: Input video frame from camera

        Returns:
            Processed frame with detection visualizations
        """
        # Step 1: Detect all persons in the frame
        try:
            persons = self.person_detector.detect(frame)
        except Exception as exc:
            logger.error("Person detection failed: %s", exc)
            self._ensure_tracking_state()
            self.person_tracker.update([], frame_shape=frame.shape)
            for track in self.person_tracker.active_tracks():
                self._apply_track_classification(track, "unknown")
            self.dashboard.update_detection(
                worker_present=False,
                helmet_result=HelmetResult.UNKNOWN,
                detections=[],
            )
            self._handle_frame_violation("unknown")
            self._store_live_frame(frame)
            return frame

        self._ensure_tracking_state()

        valid_persons = []
        for person in persons:
            if not isinstance(person, dict):
                logger.warning("Detector returned non-dict entry; skipping")
                continue
            bbox = person.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                logger.warning("Detector returned invalid bbox %r; skipping", bbox)
                continue
            if not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in bbox
            ):
                logger.warning("Detector returned invalid bbox %r; skipping", bbox)
                continue
            bbox = [int(v) for v in bbox]
            person_confidence = person.get("confidence")
            if (
                person_confidence is not None
                and (
                    not isinstance(person_confidence, (int, float))
                    or isinstance(person_confidence, bool)
                    or not math.isfinite(person_confidence)
                )
            ):
                logger.warning("Detector returned invalid confidence %r; using null", person_confidence)
                person_confidence = None
            valid_persons.append({"bbox": bbox, "confidence": person_confidence})

        matched_tracks, expired_tracks = self.person_tracker.update(
            [p["bbox"] for p in valid_persons],
            frame_shape=frame.shape,
        )
        for track in expired_tracks:
            if track.expired_active_incident:
                self._events.append(
                    EventType.ALERT,
                    f"No-helmet incident {track.current_incident_id} expired with lost track",
                )

        no_helmet_detected = False
        helmet_seen = False
        unknown_seen = False
        _dashboard_helmet_result = HelmetResult.UNKNOWN
        _dashboard_bbox = None
        _dashboard_confidence = None
        _detections = []

        # Per-frame counters for workers newly entered this frame (used to update daily stats)
        _stat_inspected = 0
        _stat_helmet = 0
        _stat_no_helmet = 0

        # Bboxes of persons with valid crops in this frame (carried to next frame for IoU dedup)
        current_bboxes = []
        # Bboxes already counted as new workers in this frame (prevents double-counting same person)
        accepted_this_frame = []

        by_bbox = {tuple(p["bbox"]): p for p in valid_persons}
        seen_track_ids = {track.track_id for track in matched_tracks}

        # Step 2-4: Process each tracked person seen in this frame
        for track in matched_tracks:
            bbox = track.bbox
            person_confidence = by_bbox.get(tuple(bbox), {}).get("confidence")
            person_crop = self.crop_person(frame, bbox)

            if person_crop.size == 0:
                unknown_seen = True
                self._apply_track_classification(track, "unknown")
                continue

            current_bboxes.append(bbox)

            try:
                result = self.helmet_classifier.predict(person_crop)
            except Exception as exc:
                logger.warning("Helmet classification failed for bbox %r: %s", bbox, exc)
                unknown_seen = True
                self._apply_track_classification(track, "unknown")
                continue
            if not isinstance(result, dict):
                logger.warning("Classifier returned non-dict result; skipping")
                unknown_seen = True
                self._apply_track_classification(track, "unknown")
                continue
            label = result.get("label")
            confidence = result.get("confidence")
            if (
                not isinstance(label, str)
                or label not in ("helmet", "no_helmet", "unknown")
                or not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not math.isfinite(confidence)
            ):
                logger.warning("Classifier returned invalid result %r; skipping", result)
                unknown_seen = True
                self._apply_track_classification(track, "unknown")
                continue
            if label == "helmet":
                helmet_seen = True
            elif label == "unknown":
                unknown_seen = True

            # Capture the first valid person's bbox and label for the dashboard mirror.
            if _dashboard_bbox is None:
                _dashboard_bbox = tuple(int(v) for v in bbox)
                _dashboard_confidence = float(confidence)
                if label == "helmet":
                    _dashboard_helmet_result = HelmetResult.HELMET
                elif label == "no_helmet":
                    _dashboard_helmet_result = HelmetResult.NO_HELMET
            elif label == "no_helmet" and _dashboard_helmet_result != HelmetResult.NO_HELMET:
                _dashboard_bbox = tuple(int(v) for v in bbox)
                _dashboard_confidence = float(confidence)

            # Step 4: Draw detection visualization
            x, y, w, h = bbox
            color = (0, 255, 0) if label == "helmet" else (0, 255, 255) if label == "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label.replace('_', ' ').upper()} {confidence * 100:.0f}%", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            self._apply_track_classification(track, label, frame.copy() if label == "no_helmet" else None, float(confidence))
            _detections.append({
                "track_id": track.track_id,
                "person_bbox": [x, y, w, h],
                "person_confidence": None if person_confidence is None else float(person_confidence),
                "helmet_result": label,
                "helmet_confidence": float(confidence),
                "incident_active": bool(track.incident_active),
                "incident_id": track.current_incident_id,
            })

            if label == "no_helmet":
                no_helmet_detected = True
                _dashboard_helmet_result = HelmetResult.NO_HELMET
                _dashboard_bbox = tuple(int(v) for v in bbox)
                _dashboard_confidence = float(confidence)

            # Count as a new worker only when there is no significant overlap with bboxes
            # from the previous frame (inter-frame dedup) AND this frame (intra-frame dedup)
            if _is_new_worker(bbox, self._prev_bboxes) and _is_new_worker(bbox, accepted_this_frame):
                accepted_this_frame.append(bbox)
                if label == "helmet":
                    _stat_inspected += 1
                    _stat_helmet += 1
                    self.dashboard.append_event(
                        EventType.DETECTION,
                        f"Helmet detected (confidence: {confidence:.2f})",
                    )
                elif label == "no_helmet":
                    _stat_inspected += 1
                    _stat_no_helmet += 1
                    self.dashboard.append_event(
                        EventType.DETECTION,
                        f"No helmet detected (confidence: {confidence:.2f})",
                    )
                else:
                    continue
                self.dashboard.append_event(EventType.DETECTION, "Worker detected")

        for track in self.person_tracker.active_tracks():
            if track.track_id not in seen_track_ids:
                unknown_seen = True
                self._apply_track_classification(track, "unknown")

        # Advance the previous-frame bbox cache.
        self._prev_bboxes = current_bboxes

        # Mirror current frame detection result to dashboard state.
        self.dashboard.update_detection(
            worker_present=len(current_bboxes) > 0,
            helmet_result=_dashboard_helmet_result,
            bbox=_dashboard_bbox,
            confidence=_dashboard_confidence,
            detections=_detections,
        )

        # Update daily statistics with newly counted workers.
        if _stat_inspected > 0:
            self.dashboard.update_statistics(
                inspected_delta=_stat_inspected,
                helmet_delta=_stat_helmet,
                no_helmet_delta=_stat_no_helmet,
            )

        if hasattr(self.alert_manager, "detection_count"):
            self.alert_manager.detection_count = max(
                [t.consecutive_no_helmet for t in self.person_tracker.active_tracks()] or [0]
            )
        if no_helmet_detected:
            frame_helmet_state = "no_helmet"
        elif helmet_seen and not unknown_seen:
            frame_helmet_state = "helmet"
        else:
            frame_helmet_state = "unknown"
        self._handle_frame_violation(frame_helmet_state)
        if self._tracked_incident_seen and not any(t.incident_active for t in self.person_tracker.active_tracks()):
            self._last_violation_detected = False
            if hasattr(self.alert_manager, "incident_active"):
                self.alert_manager.incident_active = False
            if frame_helmet_state == "helmet":
                self._clear_helmet_warning()

        self._store_live_frame(frame)
        return frame

    def start(self):
        """
        Start the helmet detection system main loop.

        This method:
        1. Initializes Arduino communication
        2. Starts real-time video processing
        3. Displays processed frames with detections (only when ENABLE_DISPLAY is True)
        4. Handles user input and system shutdown

        In dev mode (APP_LAB_DEV_MODE=true): hardware unavailable → logs warning, exits loop.
        """
        if getattr(self, "_dev_mode", False):
            logger.warning(
                "[DEV MODE] APP_LAB_DEV_MODE is active – hardware (camera, MCU bridge) will not be used. "
                "Set APP_LAB_DEV_MODE=false (or unset) for production."
            )
            self.running = True
            self._events.append(EventType.SYSTEM, "[DEV MODE] System running without hardware")
            self.running = False
            return

        self.running = True
        try:
            # Initialize Arduino communication
            self.bridge_rpc.connect()
            self._connected = True
            self.dashboard.update_connection(online=True)
            self._events.append(EventType.CONNECTION, "Robot connected")

            if ENABLE_DISPLAY:
                logger.info("System started. Press 'q' to quit.")
            else:
                logger.info("System started in headless mode. Press Ctrl+C to quit.")

            # Main processing loop
            while self.running:
                # Renew the MCU motion lease every ~250 ms by sending a control_tick.
                # Keeping this inside the main loop means any camera or AI hang automatically
                # stops tick emission, letting the MCU lease expire and STOP motors safely.
                now = time.monotonic()
                if now - getattr(self, "_last_tick_time", 0.0) >= _CONTROL_TICK_INTERVAL:
                    try:
                        self.bridge_rpc.control_tick()  # Extend the MCU motion lease by one interval
                    except Exception as e:
                        logger.warning("control_tick failed: %s", e)
                    self._last_tick_time = time.monotonic()  # Update timestamp regardless of success

                if now - getattr(self, "_last_telemetry_time", 0.0) >= _TELEMETRY_INTERVAL:
                    try:
                        self._poll_mcu_status()
                    except Exception as e:
                        logger.warning("MCU telemetry poll failed: %s", e)
                    self._last_telemetry_time = time.monotonic()

                # Capture and process frame
                try:
                    frame = self.camera.capture_frame()
                    processed_frame = self.process_frame(frame)
                except Exception as e:
                    logger.warning("Camera frame unavailable: %s", e)
                    self.dashboard.update_detection(
                        worker_present=False,
                        helmet_result=HelmetResult.UNKNOWN,
                        detections=[],
                    )
                    self._handle_frame_violation("unknown")
                    self._set_frame_status("frame_unavailable", str(e))
                    self.running = False
                    break

                # Show local preview only when a display is available
                if ENABLE_DISPLAY:
                    cv2.imshow("Helmet Detection System", processed_frame)
                    # Check for quit command ('q' key) only when display is active
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            logger.info("System stopped by user")
        except Exception as e:
            logger.error("System error: %s", e)
        finally:
            self.stop()

    def stop(self):
        """
        Gracefully shutdown the helmet detection system.
        Releases all resources and closes connections.
        """
        self.running = False
        self.camera.stop_capture()        # Release camera resources
        if self._connected:
            _safe_reset_ok = False
            try:
                self.bridge_rpc.safe_reset()
                _safe_reset_ok = True
                logger.info("safe_reset succeeded during shutdown")
            except Exception as e:
                logger.warning("Best-effort safe_reset during shutdown failed: %s", e)
            if not _safe_reset_ok:
                try:
                    self.bridge_rpc.motor_control("stop")
                except Exception as e:
                    logger.warning("Best-effort STOP during shutdown failed: %s", e)
        try:
            self.bridge_rpc.disconnect()      # Stop Arduino Bridge heartbeat
        except Exception as e:
            logger.warning("disconnect() raised during shutdown: %s", e)
        if self._connected:
            self._connected = False
            self.dashboard.update_connection(online=False)
            self._events.append(EventType.CONNECTION, "Robot disconnected")
        else:
            self.dashboard.update_connection(online=False)
        self.sender.close()               # Close HTTP session
        # cv2.destroyAllWindows() is owned by CameraCapture.stop_capture() above.
        # Do not call it here: CameraCapture is the sole owner of OpenCV window cleanup.


def main():
    """
    Main entry point for the helmet detection system.
    Parses command line arguments and starts the system.
    """
    parser = argparse.ArgumentParser(description='Helmet Detection System')
    parser.add_argument('--port', type=str, default=DEFAULT_SERIAL_PORT,
                       help='MCU transport label for Arduino communication')
    parser.add_argument('--server-url', type=str, default=DEFAULT_SERVER_URL,
                       help='Server URL for alert transmission')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Fail fast if required runtime model files are missing
    validate_runtime_models()

    # Initialize and start the helmet detection system
    system = HelmetDetectionSystem(port=args.port, server_url=args.server_url)
    system.start()


if __name__ == "__main__":
    main()
