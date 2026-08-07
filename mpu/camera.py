"""Camera Capture Module

Wraps OpenCV VideoCapture to provide a minimal, exception-safe interface
for grabbing BGR frames from a USB/built-in camera.

Frame lifecycle:
    CameraCapture.__init__  ->  _initialize_camera()  opens the device
    capture_frame()         ->  reads and returns one BGR ndarray
    stop_capture()          ->  releases the device and closes OpenCV windows
    __del__()               ->  calls stop_capture() as a fallback guard

Usage:
    camera = CameraCapture(camera_index=0)
    frame = camera.capture_frame()   # np.ndarray (H, W, 3) uint8, BGR
    camera.stop_capture()
"""

import cv2
import numpy as np
import logging
from PIL import Image
from .config import CAMERA_WIDTH, CAMERA_HEIGHT, DEFAULT_CAMERA_INDEX

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Real-time camera capture system for helmet detection.
    Captures frames from USB camera for external processing.
    """

    def __init__(self, camera_index=DEFAULT_CAMERA_INDEX):
        """
        Initialize camera capture system.

        Args:
            camera_index (int): Camera device index (default: DEFAULT_CAMERA_INDEX from config)
        """
        self.camera_index = camera_index  # OS-assigned camera device index (0 = first device)
        self.cap = None                    # cv2.VideoCapture handle; assigned in _initialize_camera()
        self.is_running = False            # Caller sets this True to keep an external capture loop alive
        self._initialize_camera()

    def _initialize_camera(self):
        """
        Initialize OpenCV camera capture with specified settings.
        Sets camera resolution to 640x480 for optimal performance.

        Raises:
            RuntimeError: If camera initialization fails
        """
        logger.info("Opening camera device index %d (requested %dx%d)",
                    self.camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
        try:
            # Create VideoCapture object for the specified camera index
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error("Failed to open camera device index %d", self.camera_index)
                raise RuntimeError(f"Failed to open camera with index {self.camera_index}")

            # Request the target resolution; drivers may silently select the nearest supported size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            # Read back the resolution the driver actually accepted for diagnostic logging
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("Camera device %d opened successfully (actual resolution %dx%d)",
                        self.camera_index, actual_w, actual_h)
        except RuntimeError:
            raise
        except Exception as e:
            logger.error("Camera initialization failed for device %d: %s", self.camera_index, e)
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

        # Grab and decode one frame; ret is False on hardware/buffer errors
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        return frame

    def stop_capture(self):
        """
        Stop camera capture and cleanup resources.
        Releases camera and closes OpenCV windows.
        """
        self.is_running = False      # Signal any external loop to exit
        if self.cap:
            self.cap.release()        # Return the camera device to the OS
        cv2.destroyAllWindows()       # Close every OpenCV display window

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
