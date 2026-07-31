"""
Helmet Detection System - Main Application

This is the main application file for a real-time helmet detection and safety monitoring system.
The system integrates computer vision, Arduino communication, and alert transmission to provide
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
- Serial communication with Arduino
- HTTP alert transmission
"""

import cv2
import numpy as np
import argparse
import logging
import time
from mpu.camera import CameraCapture
from mpu.detector import PersonDetector
from mpu.classifier import HelmetClassifier
from mpu.alert_manager import AlertManager
from mpu.bridge_rpc import BridgeRPC
from mpu.sender import Sender
from mpu.config import DEFAULT_SERIAL_PORT, DEFAULT_SERVER_URL, ENABLE_DISPLAY, validate_runtime_models
from mpu.dashboard_state import DashboardState, EventType, HelmetResult

logger = logging.getLogger(__name__)

_WORKER_IOU_THRESHOLD = 0.3
_EVENT_SUPPRESS_SECONDS = 10.0


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
        suppress_key = message if key is None else key
        now = time.monotonic()
        if now - self._last_emitted.get(suppress_key, 0.0) < self._suppress_seconds:
            return False
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

    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy

    if iw <= 0 or ih <= 0:
        return 0.0

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
            port: Serial port for Arduino communication (default from config)
            server_url: URL for remote alert transmission (default from config)
        """
        # Initialize computer vision components
        self.camera = CameraCapture()                    # Camera capture system
        self.person_detector = PersonDetector()          # Person detection AI model
        self.helmet_classifier = HelmetClassifier()      # Helmet classification AI model

        # Initialize communication components
        self.sender = Sender(server_url)                 # HTTP alert transmission
        self.bridge_rpc = BridgeRPC(port)               # Arduino serial communication

        # Initialize alert management with callback for hardware alerts
        self.alert_manager = AlertManager(callback=self.on_no_helmet_alert)

        # System state
        self.running = False
        self.alert_hardware_active = False
        self._connected = False  # True only after a successful connect()

        # Dashboard state mirror
        self.dashboard = DashboardState()

        # Event suppressor wrapping the dashboard event queue.
        self._events = _EventSuppressor(self.dashboard)

        # Bboxes from the previous frame used for new-worker detection.
        self._prev_bboxes: list = []

    def _send_alert_commands(self, led_color, buzzer_state, retries=1):
        """
        Send alert commands to Arduino with retry mechanism.

        Args:
            led_color: LED color command
            buzzer_state: Buzzer state command
            retries: Number of retry attempts (default: 1)
        """
        for attempt in range(retries + 1):
            try:
                self.bridge_rpc.led_control(led_color)
                self.bridge_rpc.buzzer_control(buzzer_state)
                return True
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        "Arduino command failed, retrying... (attempt %s/%s)",
                        attempt + 1,
                        retries + 1,
                    )
                    continue
                else:
                    msg = f"Arduino communication failed: {e}"
                    logger.error("Arduino communication failed after %s attempts: %s", retries + 1, e)
                    self._events.append(EventType.SYSTEM, msg)
                    return False

    def on_no_helmet_alert(self):
        """
        Callback function triggered when a helmet violation alert is needed.
        Activates Arduino-based LED and buzzer alerts.
        """
        if self._send_alert_commands("red", "on"):
            self.alert_hardware_active = True
            self._events.append(EventType.ALERT, "No-helmet alert triggered")

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
        persons = self.person_detector.detect(frame)

        # Track detection status across all persons
        no_helmet_detected = False
        _dashboard_helmet_result = HelmetResult.UNKNOWN
        _dashboard_bbox = None

        # Statistics accumulators for new-worker entries this frame.
        _stat_inspected = 0
        _stat_helmet = 0
        _stat_no_helmet = 0

        # Bboxes of persons with valid crops in this frame (for next-frame comparison).
        current_bboxes = []
        # Bboxes already accepted as new workers in this frame (same-frame duplicate guard).
        accepted_this_frame = []

        # Step 2-4: Process each detected person
        for person in persons:
            bbox = person["bbox"]
            person_crop = self.crop_person(frame, bbox)

            # Skip invalid crops (empty regions)
            if person_crop.size == 0:
                continue

            current_bboxes.append(bbox)

            # Step 3: Classify helmet wearing status
            result = self.helmet_classifier.predict(person_crop)
            label = result["label"]
            confidence = result["confidence"]

            # Capture the first valid person's bbox and label for the dashboard mirror.
            if _dashboard_bbox is None:
                _dashboard_bbox = tuple(int(v) for v in bbox)
                _dashboard_helmet_result = (
                    HelmetResult.HELMET if label == "helmet" else HelmetResult.NO_HELMET
                )

            # Step 4: Draw detection visualization
            x, y, w, h = bbox
            color = (0, 255, 0) if label == "helmet" else (0, 0, 255)  # Green for helmet, red for no helmet
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Step 5: Track overall detection status and send alerts
            if label != "helmet":
                no_helmet_detected = True
                _dashboard_helmet_result = HelmetResult.NO_HELMET
                try:
                    # Send alert with frame capture for remote monitoring
                    self.sender.send_alert(frame, label, confidence)
                except Exception as e:
                    logger.error("Failed to send alert: %s", e)

            # Count only when this bbox has no significant overlap with any bbox
            # from the previous frame OR already accepted in the current frame.
            if _is_new_worker(bbox, self._prev_bboxes) and _is_new_worker(bbox, accepted_this_frame):
                accepted_this_frame.append(bbox)
                _stat_inspected += 1
                if label == "helmet":
                    _stat_helmet += 1
                    self.dashboard.append_event(
                        EventType.DETECTION,
                        f"Helmet detected (confidence: {confidence:.2f})",
                    )
                else:
                    _stat_no_helmet += 1
                    self.dashboard.append_event(
                        EventType.DETECTION,
                        f"No helmet detected (confidence: {confidence:.2f})",
                    )
                self.dashboard.append_event(EventType.DETECTION, "Worker detected")

        # Advance the previous-frame bbox cache.
        self._prev_bboxes = current_bboxes

        # Mirror current frame detection result to dashboard state.
        self.dashboard.update_detection(
            worker_present=len(persons) > 0,
            helmet_result=_dashboard_helmet_result,
            bbox=_dashboard_bbox,
        )

        # Update daily statistics with newly counted workers.
        if _stat_inspected > 0:
            self.dashboard.update_statistics(
                inspected_delta=_stat_inspected,
                helmet_delta=_stat_helmet,
                no_helmet_delta=_stat_no_helmet,
            )

        # Step 6: Update alert manager and hardware status
        self.alert_manager.on_detection(no_helmet_detected)

        # Turn off alerts only when active hardware alert state returns to safe/no-person.
        if self.alert_hardware_active and not no_helmet_detected:
            if self._send_alert_commands("off", "off"):
                self.alert_hardware_active = False

        return frame

    def start(self):
        """
        Start the helmet detection system main loop.

        This method:
        1. Initializes Arduino communication
        2. Starts real-time video processing
        3. Displays processed frames with detections (only when ENABLE_DISPLAY is True)
        4. Handles user input and system shutdown
        """
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
                # Capture and process frame
                frame = self.camera.capture_frame()
                processed_frame = self.process_frame(frame)

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
        self.bridge_rpc.disconnect()      # Close Arduino serial connection
        if self._connected:
            self._connected = False
            self.dashboard.update_connection(online=False)
            self._events.append(EventType.CONNECTION, "Robot disconnected")
        else:
            self.dashboard.update_connection(online=False)
        self.sender.close()               # Close HTTP session
        # destroyAllWindows is a no-op when no windows were opened (headless mode)
        cv2.destroyAllWindows()


def main():
    """
    Main entry point for the helmet detection system.
    Parses command line arguments and starts the system.
    """
    parser = argparse.ArgumentParser(description='Helmet Detection System')
    parser.add_argument('--port', type=str, default=DEFAULT_SERIAL_PORT,
                       help='Serial port for Arduino communication')
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
