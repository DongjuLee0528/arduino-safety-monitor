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

    def __init__(self, port: str = DEFAULT_SERIAL_PORT,
                 server_url: str = DEFAULT_SERVER_URL):

        # Computer Vision
        self.camera = CameraCapture()
        self.person_detector = PersonDetector()
        self.helmet_classifier = HelmetClassifier()

        # Communication
        self.sender = Sender(server_url)
        self.bridge_rpc = BridgeRPC(port)

        # TODO 6 해결
        self.alert_manager = AlertManager(
            callback=self.on_no_helmet_alert
        )

        self.running = False
        self.alert_hardware_active = False


    def _send_alert_commands(self, led_color, buzzer_state, retries=1):

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
                        retries + 1
                    )

                else:
                    logger.error(
                        "Arduino communication failed after %s attempts: %s",
                        retries + 1,
                        e
                    )
                    return False



    def on_no_helmet_alert(self):

        if self._send_alert_commands("red", "on"):

            # TODO 7 해결
            self.alert_hardware_active = True



    def crop_person(self, frame, bbox):

        x, y, w, h = bbox

        x = max(0, x)
        y = max(0, y)

        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)

        return frame[y:y+h, x:x+w]



    def process_frame(self, frame):

        persons = self.person_detector.detect(frame)

        no_helmet_detected = False


        for person in persons:

            bbox = person["bbox"]

            person_crop = self.crop_person(frame, bbox)


            if person_crop.size == 0:
                continue


            result = self.helmet_classifier.predict(person_crop)

            label = result["label"]
            confidence = result["confidence"]


            x, y, w, h = bbox


            # TODO 8 해결
            if label != "helmet":
                color = (0, 255, 0)

            else:
                color = (0, 0, 255)


            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )


            cv2.putText(
                frame,
                f"{label}: {confidence:.2f}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )


            # TODO 9 해결
            if label != "helmet":

                no_helmet_detected = True

                try:
                    self.sender.send_alert(
                        {
                            "type": "no_helmet",
                            "confidence": confidence
                        }
                    )

                except Exception as e:

                    logger.error(
                        "Alert sending failed: %s",
                        e
                    )


        self.alert_manager.on_detection(
            no_helmet_detected
        )


        # TODO 10 해결
        if self.alert_hardware_active and not no_helmet_detected:

            try:
                self._send_alert_commands(
                    "off",
                    "off"
                )

                self.alert_hardware_active = False

            except Exception as e:

                logger.error(
                    "Failed to turn off alert: %s",
                    e
                )


        return frame



    def start(self):

        self.running = True

        try:

            # TODO 15 해결
            self.bridge_rpc.connect()

            logger.info(
                "System started. Press 'q' to quit."
            )


            while self.running:


                # TODO 11 해결
                frame = self.camera.capture_frame()


                processed_frame = self.process_frame(frame)


                cv2.imshow(
                    "Helmet Detection System",
                    processed_frame
                )


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break



        except KeyboardInterrupt:

            logger.info(
                "System stopped by user"
            )


        except Exception as e:

            logger.error(
                "System error: %s",
                e
            )


        finally:

            self.stop()



    def stop(self):

        self.running = False

        self.camera.stop_capture()

        self.bridge_rpc.disconnect()

        self.sender.close()

        cv2.destroyAllWindows()



def main():

    parser = argparse.ArgumentParser(
        description="Helmet Detection System"
    )


    parser.add_argument(
        '--port',
        type=str,
        default=DEFAULT_SERIAL_PORT,
        help='Serial port for Arduino communication'
    )


    parser.add_argument(
        '--server-url',
        type=str,
        default=DEFAULT_SERVER_URL,
        help='Server URL for alert transmission'
    )


    args = parser.parse_args()


    logging.basicConfig(
        level=logging.INFO
    )


    system = HelmetDetectionSystem(
        port=args.port,
        server_url=args.server_url
    )


    system.start()



if __name__ == "__main__":
    main()