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
import logging
from typing import List, Dict
import os
from mpu.config import MOBILENET_SSD_PATH, DETECTOR_CONFIDENCE_THRESHOLD, PERSON_CLASS_ID

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Person detection using MobileNet SSD ONNX model.

    This class provides efficient person detection for preprocessing
    frames before helmet classification. Uses COCO-trained MobileNet SSD
    with person class ID 1 (COCO).
    """
    def __init__(self, confidence_threshold: float = DETECTOR_CONFIDENCE_THRESHOLD):
        """
        Initialize person detector with MobileNet SSD model.

        Args:
            confidence_threshold: Minimum confidence score for person detection (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold  # Detection confidence threshold

        # Load ONNX model for person detection
        if not os.path.exists(MOBILENET_SSD_PATH):
            raise FileNotFoundError(f"Person detector model file not found: {MOBILENET_SSD_PATH}")

        self.session = ort.InferenceSession(MOBILENET_SSD_PATH)
        self.input_name = self.session.get_inputs()[0].name

        # COCO dataset class ID for person (1 = person in COCO)
        self.person_class_id = PERSON_CLASS_ID

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
        if frame is None:
            return []

        if not isinstance(frame, np.ndarray):
            raise ValueError(f"frame must be a numpy.ndarray, got {type(frame).__name__}")
        if frame.ndim != 3:
            raise ValueError(f"frame must be a 3-dimensional array (H, W, C), got ndim={frame.ndim}")
        if frame.shape[2] != 3:
            raise ValueError(f"frame must have 3 channels, got {frame.shape[2]}")

        if frame.dtype != np.uint8:
            if not np.issubdtype(frame.dtype, np.number):
                raise ValueError(f"frame dtype must be numeric, got {frame.dtype}")
            # Accept both normalized float frames and raw numeric frames.
            if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        h, w = frame.shape[:2]  # Get frame dimensions

        # Preprocess: BGR -> RGB, keep uint8, add batch dim -> (1, H, W, 3)
        # Note: ssd_mobilenet_v1_12.onnx expects uint8 [0,255] input (TF SavedModel
        # export convention), NOT float32 [0,1].  Do not divide by 255 here.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0)  # (1, H, W, 3) uint8

        # Run inference using ONNX Runtime
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Unpack outputs — ONNX export of ssd_mobilenet_v1_12 always yields 4 outputs
        # in this fixed order regardless of how many detections are found:
        boxes = outputs[0][0]           # (100, 4) float32 — normalized [top, left, bottom, right] in [0,1]
        classes = outputs[1][0]         # (100,)   float32 — COCO class IDs (person = 1)
        scores = outputs[2][0]          # (100,)   float32 — detection confidence scores
        num_detections = max(0, int(outputs[3][0]))  # Scalar: how many of the 100 slots are valid

        # Guard against num_detections exceeding any actual output array length.
        num_detections = min(num_detections, len(boxes), len(classes), len(scores))

        results = []
        for i in range(num_detections):
            # Skip detection if class, confidence, or bbox contains NaN/Inf
            if not np.isfinite(classes[i]) or not np.isfinite(scores[i]):
                continue
            if not np.all(np.isfinite(boxes[i])):
                continue

            class_id = int(classes[i])
            confidence = float(scores[i])

            # Filter for person detections above confidence threshold
            if class_id != self.person_class_id or confidence < self.confidence_threshold:
                continue

            # Clip normalized coords to [0, 1] then convert to pixel coords
            top, left, bottom, right = boxes[i]
            top    = float(np.clip(top,    0.0, 1.0))
            left   = float(np.clip(left,   0.0, 1.0))
            bottom = float(np.clip(bottom, 0.0, 1.0))
            right  = float(np.clip(right,  0.0, 1.0))

            x1 = int(left   * w)
            y1 = int(top    * h)
            x2 = int(right  * w)
            y2 = int(bottom * h)

            # Clip to frame boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Convert to [x, y, width, height] format
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Skip degenerate boxes
            if bbox_w <= 0 or bbox_h <= 0:
                continue

            results.append({
                "bbox": [x1, y1, bbox_w, bbox_h],
                "confidence": confidence
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
        logging.basicConfig(level=logging.INFO)
        logger.error("Could not open camera")
        exit()

    logging.basicConfig(level=logging.INFO)
    logger.info("Person detection started. Press 'q' to quit.")

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
