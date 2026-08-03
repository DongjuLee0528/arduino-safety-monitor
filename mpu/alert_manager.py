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
    """

    def __init__(
        self,
        threshold: int = DEFAULT_DETECTION_THRESHOLD,
        cooldown: float = DEFAULT_COOLDOWN_TIME,
        callback: Optional[Callable] = None,
    ):
        self.threshold = threshold
        self.cooldown = cooldown
        self.callback = callback
        self.detection_count = 0
        self.last_alert_time = 0.0

    def on_detection(self, detected: bool):
        """
        Process detection result and manage alert triggering.
        """

        if detected:
            # TODO 1 해결: 연속 감지 횟수 증가
            self.detection_count += 1

            # Check if threshold is reached and cooldown period has passed
            if self.detection_count >= self.threshold and self._can_alert():
                self._trigger_alert()

                # TODO 2 해결: 알림 발생 후 카운터 초기화
                self.detection_count = 0

                self.last_alert_time = time.time()

        else:
            # TODO 3 해결: 미감지 시 카운터 초기화
            self.detection_count = 0

    def _can_alert(self) -> bool:
        """
        Check if enough time has passed since last alert.
        """

        current_time = time.time()

        # TODO 4 해결: 쿨다운 시간 비교
        return (current_time - self.last_alert_time) >= self.cooldown

    def _trigger_alert(self):
        """
        Trigger alert and execute callback.
        """

        logger.warning("ALERT: No helmet detected!")

        # TODO 5 해결: callback 존재 시 실행
        if self.callback is not None:
            self.callback()


if __name__ == "__main__":
    """
    Test script for AlertManager functionality.
    """

    alert_manager = AlertManager()

    test_detections = [True, True, True, False, True, True, True, True]

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing AlertManager with detection sequence...")

    for i, detection in enumerate(test_detections):
        logger.info("Detection %s: %s", i + 1, detection)
        alert_manager.on_detection(detection)
        time.sleep(1)