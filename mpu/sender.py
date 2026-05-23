import requests
import base64
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, Union
import time


class Sender:
    def __init__(self, server_url: str = "http://localhost:3000/api/alert"):
        self.server_url = server_url
        self.session = requests.Session()

    def encode_image(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

    def send_alert(self, image: np.ndarray, label: str, confidence: float, retries: int = 1) -> bool:
        timestamp = datetime.now().isoformat()
        image_base64 = self.encode_image(image)

        payload = {
            "image": image_base64,
            "timestamp": timestamp,
            "detection": {
                "label": label,
                "confidence": confidence
            }
        }

        for attempt in range(retries + 1):
            try:
                response = self.session.post(
                    self.server_url,
                    json=payload,
                    timeout=10
                )

                if response.status_code == 200:
                    return True
                else:
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text}")

            except Exception as e:
                if attempt < retries:
                    time.sleep(1)
                    continue
                else:
                    raise RuntimeError(f"Failed to send alert after {retries + 1} attempts: {e}")

        return False

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "TEST NO HELMET", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    try:
        with Sender() as sender:
            result = sender.send_alert(test_image, "no_helmet", 0.95)
            if result:
                print("Alert sent successfully")
            else:
                print("Failed to send alert")
    except Exception as e:
        print(f"Error: {e}")
