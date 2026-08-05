"""
Alert Manager for Helmet Detection System

This module provides intelligent alert management with configurable thresholds
and cooldown periods to prevent alert spam while ensuring safety violations
are properly reported.

Key Features:
- Threshold-based alerting (requires multiple consecutive detections)
- Cooldown periods between alerts to prevent spam
- Customizable callback system for alert actions
- Simple detection count tracking

Usage:
    alert_manager = AlertManager(threshold=3, cooldown=5.0, callback=my_alert_function)
    alert_manager.on_detection(no_helmet_detected)  # Call this for each frame
"""

import time
import logging
from typing import Callable, Optional
from mpu.config import DEFAULT_DETECTION_THRESHOLD, DEFAULT_COOLDOWN_TIME

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alert triggering with threshold and cooldown mechanisms.

    This class prevents false alarms and alert spam by requiring multiple
    consecutive detections before triggering an alert, and enforcing a
    cooldown period between successive alerts.
    """
    def __init__(
        self,
        threshold: int = DEFAULT_DETECTION_THRESHOLD,
        cooldown: float = DEFAULT_COOLDOWN_TIME,
        callback: Optional[Callable] = None,
    ):
        """
        Initialize the alert manager with configurable parameters.

        Args:
            threshold: Number of consecutive detections required to trigger alert
            cooldown: Minimum time (seconds) between alerts
            callback: Optional function to call when alert is triggered
        """
        self.threshold = threshold          # Number of consecutive frames required before alert fires
        self.cooldown = cooldown            # Minimum seconds that must pass between two alerts
        self.callback = callback            # Optional function called when an alert is triggered
        self.detection_count = 0            # Running count of consecutive no-helmet detections
        self.last_alert_time = 0.0          # time.time() value when the last alert was triggered

    def on_detection(self, detected: bool):
        """
        Process detection result and manage alert triggering.

        Args:
            detected: True if safety violation (no helmet) was detected in this frame
        """
        if detected:
            self.detection_count += 1  # Increment consecutive violation count
            # Fire alert only when both the frame threshold and the cooldown are satisfied
            if self.detection_count >= self.threshold and self._can_alert():
                self._trigger_alert()
                self.detection_count = 0              # Reset counter after alert fires
                self.last_alert_time = time.time()    # Record timestamp for next cooldown check
        else:
            # Any frame with no violation resets the consecutive counter
            self.detection_count = 0

    def _can_alert(self) -> bool:
        """
        Check if enough time has passed since last alert (cooldown check).

        Returns:
            True if alert can be triggered, False if still in cooldown period
        """
        current_time = time.time()
        # True when enough time has elapsed since the last alert
        return current_time - self.last_alert_time >= self.cooldown

    def _trigger_alert(self):
        """
        Trigger alert by printing message and calling callback function.
        This method is called when threshold and cooldown conditions are met.
        """
        logger.warning("ALERT: No helmet detected!")
        if self.callback:
            self.callback()                     # Execute custom alert callback


if __name__ == "__main__":
    """
    Test script for AlertManager functionality.
    Simulates detection sequence to verify threshold and cooldown behavior.
    """
    # Initialize alert manager with default settings
    alert_manager = AlertManager()

    # Test detection sequence: should trigger alert after 3rd consecutive True
    test_detections = [True, True, True, False, True, True, True, True]

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing AlertManager with detection sequence...")
    for i, detection in enumerate(test_detections):
        logger.info("Detection %s: %s", i + 1, detection)
        alert_manager.on_detection(detection)
        time.sleep(1)  # Simulate real-time processing
