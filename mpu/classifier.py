import os
import logging
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from mpu.config import ONNX_MODEL_PATH, MODEL_INPUT_SIZE

logger = logging.getLogger(__name__)


class HelmetClassifier:
    """
    Helmet detection classifier using ONNX model for real-time inference.
    Classifies images to determine helmet wearing status.
    """

    def __init__(self, model_path=ONNX_MODEL_PATH):
        """
        Initialize the helmet classifier with ONNX model.

        Args:
            model_path (str): Path to the ONNX model file
        """
        self.model_path = model_path
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
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess_image(self, image):
        """
        Preprocess input image for model inference.
        Converts to RGB, resizes to 224x224, and applies ImageNet normalization.

        Args:
            image: PIL Image or numpy array

        Returns:
            np.ndarray: Preprocessed image tensor ready for inference
        """
        # OpenCV frames arrive as BGR arrays; convert to RGB before creating a PIL image.
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model input size (MODEL_INPUT_SIZE x MODEL_INPUT_SIZE)
        image = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        image_array = np.array(image).astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        image_array = (image_array - mean) / std

        # Convert from HWC to CHW format and add batch dimension
        image_array = image_array.transpose(2, 0, 1)
        image_array = np.expand_dims(image_array, axis=0)

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

        # Run inference using ONNX runtime
        outputs = self.session.run([self.output_name], {self.input_name: processed_image})
        predictions = outputs[0][0]

        # Convert logits to probabilities using softmax
        probabilities = self._softmax(predictions)
        confidence = float(np.max(probabilities))
        predicted_class = int(np.argmax(probabilities))

        # Map class index to label (1: helmet, 0: no_helmet)
        label = "helmet" if predicted_class == 1 else "no_helmet"

        return {
            "label": label,
            "confidence": confidence
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


if __name__ == "__main__":
    """
    Main execution block for testing the helmet classifier.
    Initializes the classifier and optionally runs inference on a test image.
    """
    # Initialize the helmet classifier
    logging.basicConfig(level=logging.INFO)
    classifier = HelmetClassifier()
    logger.info("HelmetClassifier initialized successfully")
    logger.info("Model loaded from: %s", classifier.model_path)

    # Test image path for helmet detection validation
    # test_image_path = "/Users/dongjulee/Documents/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset/Images/hard_hat_workers0.png"

    # # Run inference on test image
    # try:
    #     from PIL import Image
    #     test_image = Image.open(test_image_path)
    #     result = classifier.predict(test_image)
    #     print(f"Prediction: {result}")
    # except Exception as e:
    #     print(f"Error processing test image: {e}")
