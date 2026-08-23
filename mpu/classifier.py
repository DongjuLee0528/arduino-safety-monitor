"""Helmet Classification Module

Provides ONNX-based binary classification (helmet / no_helmet) on
cropped image regions supplied by the person detector.

Pipeline:
    1. Convert BGR ndarray to RGB PIL Image
    2. Resize to MODEL_INPUT_SIZE x MODEL_INPUT_SIZE
    3. Apply ImageNet mean/std normalisation
    4. Transpose HWC -> CHW, add batch dim
    5. Run ONNX inference, apply softmax, return label + confidence

Usage:
    classifier = HelmetClassifier()
    result = classifier.predict(frame_crop)   # {'label': 'helmet', 'confidence': 0.97}
"""

import os
import logging
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from mpu.config import HELMET_ACCEPTANCE_THRESHOLD, ONNX_MODEL_PATH, MODEL_INPUT_SIZE

logger = logging.getLogger(__name__)


class HelmetClassifier:
    """
    Helmet detection classifier using ONNX model for real-time inference.
    Classifies images to determine helmet wearing status.
    """

    def __init__(self, model_path=ONNX_MODEL_PATH, helmet_threshold: float = HELMET_ACCEPTANCE_THRESHOLD):
        """
        Initialize the helmet classifier with ONNX model.

        Args:
            model_path (str): Path to the ONNX model file
        """
        self.model_path = model_path
        self.helmet_threshold = self._validate_threshold(helmet_threshold)
        self.session = None
        self._load_model()

    def _load_model(self):
        """
        Load the ONNX model and initialize inference session.
        Sets up input and output tensor names for the model.

        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Create ONNX runtime session for inference
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name   # Name of the model's input tensor
        self.output_name = self.session.get_outputs()[0].name # Name of the model's output tensor

    def _preprocess_image(self, image):
        """
        Preprocess input image for model inference.
        Converts to RGB, resizes to 224x224, and applies ImageNet normalization.

        Args:
            image: PIL Image or numpy array

        Returns:
            np.ndarray: Preprocessed image tensor ready for inference
        """
        # OpenCV frames arrive as BGR arrays; convert to RGB before creating a PIL Image
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB colour space swap
            image = Image.fromarray(image)

        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model input size (MODEL_INPUT_SIZE x MODEL_INPUT_SIZE)
        image = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        image_array = np.array(image).astype(np.float32) / 255.0  # Scale pixel values from [0,255] to [0.0,1.0]

        # Apply ImageNet normalization (mean and std per channel, RGB order)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        image_array = (image_array - mean) / std

        # Transpose from HWC (H, W, C) to CHW (C, H, W) required by PyTorch/ONNX models
        image_array = image_array.transpose(2, 0, 1)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension: (1, C, H, W)

        return image_array.astype(np.float32)

    def predict(self, image):
        """
        Predict helmet wearing status from input image.

        Args:
            image: PIL Image or numpy array

        Returns:
            dict: Prediction result with label and confidence score
                  {"label": "helmet" or "no_helmet", "confidence": float}
        """
        # Preprocess the input image
        processed_image = self._preprocess_image(image)

        # Run forward pass through the ONNX model
        outputs = self.session.run([self.output_name], {self.input_name: processed_image})
        predictions = outputs[0][0]  # Extract the single-sample output vector

        return self._classify_logits(predictions)

    def _classify_logits(self, logits):
        """
        Convert raw model logits to a labelled prediction dict.

        The model outputs two logits: index 0 = no_helmet, index 1 = helmet.
        Softmax is applied to obtain probabilities; the helmet class is chosen
        only when P(helmet) >= helmet_threshold (default 0.83), otherwise the
        label is "no_helmet" with confidence = P(no_helmet).

        Args:
            logits: Raw output vector from the ONNX model (shape (2,)).

        Returns:
            dict with keys:
                "label":              "helmet" | "no_helmet" | "unknown"
                "confidence":         Probability of the chosen class [0.0, 1.0]
                "helmet_probability": Raw P(helmet) float, or None on invalid input
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (2,) or not np.all(np.isfinite(logits)):
            return {"label": "unknown", "confidence": 0.0, "helmet_probability": None}

        probabilities = self._softmax(logits)
        if probabilities.shape != (2,) or not np.all(np.isfinite(probabilities)):
            return {"label": "unknown", "confidence": 0.0, "helmet_probability": None}

        # Class indices: 0 = no_helmet, 1 = helmet (training label order)
        helmet_probability = float(probabilities[1])
        if helmet_probability >= self.helmet_threshold:
            return {
                "label": "helmet",
                "confidence": helmet_probability,
                "helmet_probability": helmet_probability,
            }
        return {
            "label": "no_helmet",
            "confidence": float(probabilities[0]),
            "helmet_probability": helmet_probability,
        }

    def _softmax(self, x):
        """
        Apply softmax function to convert logits to probabilities.

        Args:
            x (np.ndarray): Input logits

        Returns:
            np.ndarray: Probability distribution
        """
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

    def _validate_threshold(self, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("helmet_threshold must be a finite number between 0.0 and 1.0")
        value = float(value)
        if not np.isfinite(value) or not (0.0 <= value <= 1.0):
            raise ValueError("helmet_threshold must be a finite number between 0.0 and 1.0")
        return value


if __name__ == "__main__":
    # Main execution block: initializes the classifier and optionally runs
    # inference on a test image.  Run directly to verify model loading.
    # Initialize the helmet classifier
    logging.basicConfig(level=logging.INFO)
    classifier = HelmetClassifier()
    logger.info("HelmetClassifier initialized successfully")
    logger.info("Model loaded from: %s", classifier.model_path)

    # # Run inference on test image
    # try:
    #     from PIL import Image
    #     test_image = Image.open(test_image_path)
    #     result = classifier.predict(test_image)
    #     print(f"Prediction: {result}")
    # except Exception as e:
    #     print(f"Error processing test image: {e}")
