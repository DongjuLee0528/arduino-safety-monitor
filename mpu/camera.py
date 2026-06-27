import cv2
import numpy as np
import logging
from PIL import Image
from .config import CAMERA_WIDTH, CAMERA_HEIGHT

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Real-time camera capture system for helmet detection.
    Captures frames from USB camera for external processing.
    """

    def __init__(self, camera_index=0):
        """
        Initialize camera capture system.

        Args:
            camera_index (int): Camera device index (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self._initialize_camera()

    def _initialize_camera(self):
        """
        Initialize OpenCV camera capture with specified settings.
        Sets camera resolution to 640x480 for optimal performance.

        Raises:
            RuntimeError: If camera initialization fails
        """
        try:
            # Create VideoCapture object for the specified camera index
            # TODO 12: OpenCV VideoCapture 객체를 생성하세요 (객체 생성)
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera with index {self.camera_index}")

            # Set camera resolution for consistent frame size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        except Exception as e:
            raise RuntimeError(f"Camera initialization failed: {e}")

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
        # TODO 13: 프레임 수집에 실패했을 때 예외를 발생시키세요 (if 코드 흐름)
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        return frame

    def stop_capture(self):
        """
        Stop camera capture and cleanup resources.
        Releases camera and closes OpenCV windows.
        """
        self.is_running = False
        # TODO 14: cap이 존재할 때만 release()를 호출하세요 (if 조건문)
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
    Main execution block for testing camera capture.
    Initializes camera capture system and displays frames.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        # Initialize camera capture with default settings
        camera = CameraCapture()
        logger.info("Camera capture initialized. Press 'q' to quit.")

        # Start real-time camera display
        camera.is_running = True
        while camera.is_running:
            frame = camera.capture_frame()
            cv2.imshow('Camera Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.stop_capture()
    except Exception as e:
        logger.error("Failed to start camera capture: %s", e)
