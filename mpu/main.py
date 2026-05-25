import cv2
import numpy as np
import argparse
from mpu.camera import CameraCapture
from mpu.detector import PersonDetector
from mpu.classifier import HelmetClassifier
from mpu.alert_manager import AlertManager
from mpu.bridge_rpc import BridgeRPC
from mpu.sender import Sender
from mpu.config import DEFAULT_SERIAL_PORT, DEFAULT_SERVER_URL


class HelmetDetectionSystem:
    def __init__(self, port: str = DEFAULT_SERIAL_PORT, server_url: str = DEFAULT_SERVER_URL):
        self.camera = CameraCapture()
        self.person_detector = PersonDetector()
        self.helmet_classifier = HelmetClassifier()
        self.sender = Sender(server_url)
        self.bridge_rpc = BridgeRPC(port)
        self.alert_manager = AlertManager(callback=self.on_no_helmet_alert)
        self.running = False

    def on_no_helmet_alert(self):
        try:
            self.bridge_rpc.led_control("red")
            self.bridge_rpc.buzzer_control("on")
        except Exception as e:
            print(f"RPC command failed: {e}")

    def crop_person(self, frame, bbox):
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)
        return frame[y:y+h, x:x+w]

    def process_frame(self, frame):
        persons = self.person_detector.detect(frame)

        helmet_detected = False
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
            color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if label == "helmet":
                helmet_detected = True
            else:
                no_helmet_detected = True
                try:
                    self.sender.send_alert(frame, label, confidence)
                except Exception as e:
                    print(f"Failed to send alert: {e}")

        self.alert_manager.on_detection(no_helmet_detected)

        if helmet_detected and not no_helmet_detected:
            try:
                self.bridge_rpc.led_control("off")
            except Exception as e:
                print(f"RPC command failed: {e}")

        return frame

    def start(self):
        self.running = True
        try:
            self.bridge_rpc.connect()
            print("System started. Press 'q' to quit.")

            while self.running:
                frame = self.camera.capture_frame()
                processed_frame = self.process_frame(frame)

                cv2.imshow("Helmet Detection System", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("System stopped by user")
        except Exception as e:
            print(f"System error: {e}")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.camera.stop_capture()
        self.bridge_rpc.disconnect()
        self.sender.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Helmet Detection System')
    parser.add_argument('--port', type=str, default=DEFAULT_SERIAL_PORT,
                       help='Serial port for Arduino communication')
    parser.add_argument('--server-url', type=str, default=DEFAULT_SERVER_URL,
                       help='Server URL for alert transmission')
    args = parser.parse_args()

    system = HelmetDetectionSystem(port=args.port, server_url=args.server_url)
    system.start()


if __name__ == "__main__":
    main()
