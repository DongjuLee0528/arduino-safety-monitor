import time
from typing import Callable, Optional


class AlertManager:
    def __init__(self, threshold: int = 3, cooldown: float = 5.0, callback: Optional[Callable] = None):
        self.threshold = threshold
        self.cooldown = cooldown
        self.callback = callback
        self.detection_count = 0
        self.last_alert_time = 0.0

    def on_detection(self, detected: bool):
        if detected:
            self.detection_count += 1
            if self.detection_count >= self.threshold and self._can_alert():
                self._trigger_alert()
                self.detection_count = 0
                self.last_alert_time = time.time()
        else:
            self.detection_count = 0

    def _can_alert(self) -> bool:
        current_time = time.time()
        return current_time - self.last_alert_time >= self.cooldown

    def _trigger_alert(self):
        print("ALERT: No helmet detected!")
        if self.callback:
            self.callback()


if __name__ == "__main__":
    alert_manager = AlertManager()

    test_detections = [True, True, True, False, True, True, True, True]

    for i, detection in enumerate(test_detections):
        print(f"Detection {i+1}: {detection}")
        alert_manager.on_detection(detection)
        time.sleep(1)
