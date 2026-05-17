import cv2
import numpy as np
from PIL import Image
from .classifier import HelmetClassifier


class CameraCapture:
    def __init__(self, camera_index=0, model_path="mpu/ai/models/best_model.onnx"):
        self.camera_index = camera_index
        self.cap = None
        self.classifier = None
        self.is_running = False
        self._initialize_camera()
        self._initialize_classifier(model_path)

    def _initialize_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            raise RuntimeError(f"Camera initialization failed: {e}")

    def _initialize_classifier(self, model_path):
        try:
            self.classifier = HelmetClassifier(model_path)
        except Exception as e:
            raise RuntimeError(f"Classifier initialization failed: {e}")

    def capture_frame(self):
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera is not initialized or opened")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        return frame

    def predict_frame(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            result = self.classifier.predict(pil_image)
            return result
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def process_frame(self):
        frame = self.capture_frame()
        prediction = self.predict_frame(frame)
        return frame, prediction

    def start_capture(self):
        self.is_running = True
        try:
            while self.is_running:
                frame, prediction = self.process_frame()

                label = prediction["label"]
                confidence = prediction["confidence"]

                color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
                text = f"{label}: {confidence:.2f}"

                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                cv2.imshow('Helmet Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Capture stopped by user")
        except Exception as e:
            print(f"Error during capture: {e}")
        finally:
            self.stop_capture()

    def stop_capture(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.stop_capture()


if __name__ == "__main__":
    try:
        camera = CameraCapture()
        print("Camera capture initialized. Press 'q' to quit.")
        camera.start_capture()
    except Exception as e:
        print(f"Failed to start camera capture: {e}")
