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
from mpu.camera import CameraCapture
from mpu.detector import PersonDetector
from mpu.classifier import HelmetClassifier
from mpu.alert_manager import AlertManager
from mpu.bridge_rpc import BridgeRPC
from mpu.sender import Sender
from mpu.config import DEFAULT_SERIAL_PORT, DEFAULT_SERVER_URL

logger = logging.getLogger(__name__)


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
        # TODO 6: AlertManager를 생성하세요 - 콜백 함수는 on_no_helmet_alert입니다 (객체 생성)
        self.alert_manager = AlertManager(callback=slef.on_no_helmet_alert)

        # System state
        self.running = False
        self.alert_hardware_active = False

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
                    logger.error("Arduino communication failed after %s attempts: %s", retries + 1, e)
                    return False

    def on_no_helmet_alert(self):
        """
        Callback function triggered when a helmet violation alert is needed.
        Activates Arduino-based LED and buzzer alerts.
        """
        if self._send_alert_commands("red", "on"):
            # TODO 7: 알림 하드웨어가 활성화되었음을 True로 표시하세요 (bool 변수)
            self.alert_hardware_active = True

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

        # Step 2-4: Process each detected person
        for person in persons:
            bbox = person["bbox"]
            person_crop = self.crop_person(frame, bbox)

            # Skip invalid crops (empty regions)
            if person_crop.size == 0:
                continue

            # Step 3: Classify helmet wearing status
            result = self.helmet_classifier.predict(person_crop)
            label = result["label"]
            confidence = result["confidence"]

            # Step 4: Draw detection visualization
            x, y, w, h = bbox
            # TODO 8: label이 "helmet"이면 착용 상태입니다. 올바른 라벨을 비교하세요 (문자열 비교)
            color = (0, 255, 0) if label == "helmet" else (0, 0, 255)  # Green for helmet, red for no helmet
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Step 5: Track overall detection status and send alerts
            # TODO 9: label이 "helmet"이 아닐 때 미착용 감지로 표시하세요 (부정 조건)
            if label != "helmet":
                no_helmet_detected = True
                try:
                    # Send alert with frame capture for remote monitoring
                    self.sender.send_alert(frame, label, confidence)
                except Exception as e:
                    logger.error("Failed to send alert: %s", e)

        # Step 6: Update alert manager and hardware status
        self.alert_manager.on_detection(no_helmet_detected)

        # Turn off alerts only when active hardware alert state returns to safe/no-person.
        # TODO 10: 하드웨어 알림이 활성화된 상태이고 동시에 미착용이 감지되지 않을 때 알림을 끄세요 (논리 연산자)
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
        3. Displays processed frames with detections
        4. Handles user input and system shutdown
        """
        self.running = True
        try:
            # Initialize Arduino communication
            # TODO 15: Arduino 시리얼 연결을 시작하세요 (메서드 호출)
            self.bridge_rpc.connect()
            logger.info("System started. Press 'q' to quit.")

            # Main processing loop
            while self.running:
                # Capture and process frame
                # TODO 11: 카메라에서 프레임을 수집하세요 (메서드 호출)
                frame = self.camera.capture_frame()
                processed_frame = self.process_frame(frame)

                # Display processed frame with detections
                cv2.imshow("Helmet Detection System", processed_frame)

                # Check for quit command ('q' key)
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
        self.sender.close()               # Close HTTP session
        cv2.destroyAllWindows()           # Close OpenCV windows


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

    # Initialize and start the helmet detection system
    system = HelmetDetectionSystem(port=args.port, server_url=args.server_url)
    system.start()


if __name__ == "__main__":
    main()
