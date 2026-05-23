import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict
import os


class PersonDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        model_path = os.path.join(os.path.dirname(__file__), "ai", "models", "mobilenet_ssd.onnx")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.person_class_id = 15

    def detect(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5))

        outputs = self.session.run(None, {self.input_name: blob})
        detections = outputs[0][0]

        results = []
        for detection in detections:
            class_id = int(detection[1])
            confidence = detection[2]

            if class_id == self.person_class_id and confidence >= self.confidence_threshold:
                x1, y1, x2, y2 = detection[3:7]
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)

                bbox_w = x2 - x1
                bbox_h = y2 - y1

                results.append({
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "confidence": float(confidence)
                })

        return results


if __name__ == "__main__":
    detector = PersonDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for detection in detections:
            x, y, w, h = detection["bbox"]
            confidence = detection["confidence"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()