"""
Person Detection Module

This module provides real-time person detection using MobileNet SSD model
optimized for edge deployment. It detects people in video frames and returns
bounding box coordinates for helmet classification.

Key Features:
- ONNX-based MobileNet SSD model for efficient person detection
- Configurable confidence threshold for detection filtering
- Returns standardized bounding box format [x, y, width, height]
- Optimized for real-time processing

Usage:
    detector = PersonDetector(confidence_threshold=0.5)
    persons = detector.detect(frame)
    for person in persons:
        bbox = person["bbox"]  # [x, y, width, height]
        confidence = person["confidence"]
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict
import os
from mpu.config import MOBILENET_SSD_PATH


class PersonDetector:
    """
    Person detection using MobileNet SSD ONNX model.

    This class provides efficient person detection for preprocessing
    frames before helmet classification. Uses COCO-trained MobileNet SSD
    with person class ID 15.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize person detector with MobileNet SSD model.

        Args:
            confidence_threshold: Minimum confidence score for person detection (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold  # Detection confidence threshold

        # Load ONNX model for person detection
        self.session = ort.InferenceSession(MOBILENET_SSD_PATH)
        self.input_name = self.session.get_inputs()[0].name

        # COCO dataset class ID for person (used by MobileNet SSD)
        self.person_class_id = 15

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in the input frame using MobileNet SSD.

        Args:
            frame: Input video frame in BGR format

        Returns:
            List of person detections, each containing:
            - "bbox": [x, y, width, height] bounding box coordinates
            - "confidence": Detection confidence score (0.0-1.0)
        """
        h, w = frame.shape[:2]  # Get frame dimensions

        # Preprocess frame for MobileNet SSD (300x300, normalized)
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5))

        # Run inference using ONNX Runtime
        outputs = self.session.run(None, {self.input_name: blob})
        detections = outputs[0][0]  # Extract detection results

        results = []
        for detection in detections:
            class_id = int(detection[1])     # Object class ID
            confidence = detection[2]        # Detection confidence

            # Filter for person detections above confidence threshold
            if class_id == self.person_class_id and confidence >= self.confidence_threshold:
                # Extract normalized bounding box coordinates
                x1, y1, x2, y2 = detection[3:7]

                # Convert to pixel coordinates
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)

                # Convert to [x, y, width, height] format
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                results.append({
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "confidence": float(confidence)
                })

        return results


if __name__ == "__main__":
    """
    Test script for PersonDetector functionality.
    Displays real-time person detection on camera feed.
    """
    # Initialize person detector with default confidence threshold
    detector = PersonDetector()

    # Open camera capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    print("Person detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons in current frame
        detections = detector.detect(frame)

        # Draw bounding boxes and labels for detected persons
        for detection in detections:
            x, y, w, h = detection["bbox"]
            confidence = detection["confidence"]

            # Draw green bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add confidence label
            cv2.putText(frame, f"Person: {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display frame with detections
        cv2.imshow("Person Detection", frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()