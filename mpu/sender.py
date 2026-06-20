"""
Alert Transmission Module

This module handles HTTP transmission of safety alerts to remote monitoring servers.
It encodes camera frames, packages detection data, and sends alerts with automatic
retry mechanisms for robust alert delivery.

Key Features:
- Base64 image encoding for web transmission
- JSON payload with timestamp and detection metadata
- Automatic retry mechanism with configurable attempts
- Session management for connection reuse
- Context manager support for resource cleanup

Alert Payload Structure:
{
    "image": "base64_encoded_jpeg",
    "timestamp": "2024-01-01T12:00:00.000000",
    "detection": {
        "label": "no_helmet",
        "confidence": 0.95
    }
}

Usage:
    with Sender("http://server/api/alert") as sender:
        success = sender.send_alert(frame, "no_helmet", 0.95)
"""

import requests
import base64
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Union
import time

from mpu.config import DEFAULT_SERVER_URL, HTTP_TIMEOUT, RETRY_DELAY

logger = logging.getLogger(__name__)

class Sender:
    """
    HTTP alert transmission system for safety monitoring.

    This class manages the transmission of safety violation alerts
    to remote monitoring servers via HTTP POST requests with
    image data and detection metadata.
    """
    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        """
        Initialize alert sender with server configuration.

        Args:
            server_url: HTTP endpoint URL for alert transmission
        """
        self.server_url = server_url          # Target server URL for alerts
        self.session = requests.Session()     # Reusable HTTP session for performance

    def encode_image(self, image: np.ndarray) -> str:
        """
        Encode image frame to base64 JPEG for web transmission.

        Args:
            image: OpenCV image frame in BGR format

        Returns:
            Base64-encoded JPEG string
        """
        # Encode image as JPEG and convert to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

    def send_alert(self, image: np.ndarray, label: str, confidence: float, retries: int = 1) -> bool:
        """
        Send safety alert with image and detection data to monitoring server.

        Args:
            image: Camera frame containing the safety violation
            label: Detection label (e.g., "no_helmet")
            confidence: Detection confidence score (0.0-1.0)
            retries: Number of retry attempts on failure

        Returns:
            True if alert sent successfully, False otherwise

        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Create timestamp for alert
        timestamp = datetime.now().isoformat()

        # Encode image for transmission
        image_base64 = self.encode_image(image)

        # Build JSON payload
        payload = {
            "image": image_base64,
            "timestamp": timestamp,
            "detection": {
                "label": label,
                "confidence": confidence
            }
        }

        # Attempt transmission with retry logic
        for attempt in range(retries + 1):
            try:
                # Send POST request to monitoring server
                response = self.session.post(
                    self.server_url,
                    json=payload,
                    timeout=HTTP_TIMEOUT  # Configurable timeout from config
                )

                # Check for successful response
                if response.status_code == 200:
                    return True
                else:
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text}")

            except Exception as e:
                # Retry on failure (with delay)
                if attempt < retries:
                    time.sleep(RETRY_DELAY)  # Wait before retry
                    continue
                else:
                    # All attempts failed
                    raise RuntimeError(f"Failed to send alert after {retries + 1} attempts: {e}")

        return False

    def close(self):
        """
        Close the HTTP session and cleanup resources.
        """
        self.session.close()

    def __enter__(self):
        """
        Context manager entry - return self for use in 'with' statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - cleanup resources.
        """
        self.close()


if __name__ == "__main__":
    """
    Test script for Sender functionality.
    Creates a test image and attempts to send it as an alert.
    """
    # Create test image with "NO HELMET" text
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "TEST NO HELMET", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    logging.basicConfig(level=logging.INFO)
    try:
        # Test alert transmission using context manager
        with Sender() as sender:
            logger.info("Testing alert transmission...")
            result = sender.send_alert(test_image, "no_helmet", 0.95)

            if result:
                logger.info("Alert sent successfully")
            else:
                logger.error("Failed to send alert")

    except Exception as e:
        logger.error("Error: %s", e)
