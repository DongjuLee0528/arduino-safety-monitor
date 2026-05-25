import cv2
import numpy as np
from PIL import Image
from .classifier import HelmetClassifier
from .config import CAMERA_WIDTH, CAMERA_HEIGHT


class CameraCapture:
    """
    Real-time camera capture system for helmet detection.
    Captures frames from USB camera and runs helmet classification inference.
    """

    def __init__(self, camera_index=0, model_path="mpu/ai/models/best_model.onnx"):
        """
        Initialize camera capture system with helmet classifier.

        Args:
            camera_index (int): Camera device index (default: 0)
            model_path (str): Path to the ONNX model file
        """
        self.camera_index = camera_index
        self.cap = None
        self.classifier = None
        self.is_running = False
        self._initialize_camera()
        self._initialize_classifier(model_path)

    def _initialize_camera(self):
        """
        Initialize OpenCV camera capture with specified settings.
        Sets camera resolution to 640x480 for optimal performance.

        Raises:
            RuntimeError: If camera initialization fails
        """
        try:
            # Create VideoCapture object for the specified camera index
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera with index {self.camera_index}")

            # Set camera resolution for consistent frame size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        except Exception as e:
            raise RuntimeError(f"Camera initialization failed: {e}")

    def _initialize_classifier(self, model_path):
        """
        Initialize the helmet classifier with the ONNX model.

        Args:
            model_path (str): Path to the ONNX model file

        Raises:
            RuntimeError: If classifier initialization fails
        """
        try:
            self.classifier = HelmetClassifier(model_path)
        except Exception as e:
            raise RuntimeError(f"Classifier initialization failed: {e}")

    def capture_frame(self):
        """
        Capture a single frame from the camera.

        Returns:
            np.ndarray: Captured frame in BGR format

        Raises:
            RuntimeError: If camera is not initialized or frame capture fails
        """
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera is not initialized or opened")

        # Read frame from camera
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        return frame

    def predict_frame(self, frame):
        """
        Run helmet detection inference on the captured frame.

        Args:
            frame (np.ndarray): Camera frame in BGR format

        Returns:
            dict: Prediction result with label and confidence score

        Raises:
            RuntimeError: If prediction fails
        """
        try:
            # Convert BGR to RGB for PIL Image compatibility
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Run helmet classification inference
            result = self.classifier.predict(pil_image)
            return result
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def process_frame(self):
        """
        Capture and process a single frame with helmet detection.

        Returns:
            tuple: (frame, prediction) containing the captured frame and prediction result
        """
        frame = self.capture_frame()
        prediction = self.predict_frame(frame)
        return frame, prediction

    def start_capture(self):
        """
        Start real-time camera capture with helmet detection display.
        Continuously captures frames, runs inference, and displays results.
        Press 'q' to quit the capture loop.
        """
        self.is_running = True
        try:
            while self.is_running:
                # Capture frame and run inference
                frame, prediction = self.process_frame()

                # Extract prediction results
                label = prediction["label"]
                confidence = prediction["confidence"]

                # Set color based on helmet detection (green for helmet, red for no helmet)
                color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
                text = f"{label}: {confidence:.2f}"

                # Overlay prediction text on frame
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Display frame with prediction overlay
                cv2.imshow('Helmet Detection', frame)

                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Capture stopped by user")
        except Exception as e:
            print(f"Error during capture: {e}")
        finally:
            # Ensure cleanup happens even if an error occurs
            self.stop_capture()

    def stop_capture(self):
        """
        Stop camera capture and cleanup resources.
        Releases camera and closes OpenCV windows.
        """
        self.is_running = False
        if self.cap:
            self.cap.release()  # Release camera resource
        cv2.destroyAllWindows()  # Close all OpenCV windows

    def __del__(self):
        """
        Destructor to ensure proper cleanup when object is destroyed.
        """
        self.stop_capture()


if __name__ == "__main__":
    """
    Main execution block for real-time helmet detection.
    Initializes camera capture system and starts the detection loop.
    """
    try:
        # Initialize camera capture with default settings
        camera = CameraCapture()
        print("Camera capture initialized. Press 'q' to quit.")

        # Start real-time helmet detection
        camera.start_capture()
    except Exception as e:
        print(f"Failed to start camera capture: {e}")
