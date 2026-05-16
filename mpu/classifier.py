import os
import numpy as np
from PIL import Image
import onnxruntime as ort


class HelmetClassifier:
    def __init__(self, model_path="mpu/ai/models/best_model.onnx"):
        self.model_path = model_path
        self.session = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        image_array = (image_array - mean) / std

        image_array = image_array.transpose(2, 0, 1)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array.astype(np.float32)

    def predict(self, image):
        processed_image = self._preprocess_image(image)

        outputs = self.session.run([self.output_name], {self.input_name: processed_image})
        predictions = outputs[0][0]

        probabilities = self._softmax(predictions)
        confidence = float(np.max(probabilities))
        predicted_class = int(np.argmax(probabilities))

        label = "helmet" if predicted_class == 1 else "no_helmet"

        return {
            "label": label,
            "confidence": confidence
        }

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


if __name__ == "__main__":
    classifier = HelmetClassifier()
    print("HelmetClassifier initialized successfully")
    print(f"Model loaded from: {classifier.model_path}")

    test_image_path = "/Users/dongjulee/Documents/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset/Images/hard_hat_workers0.png"

    try:
        from PIL import Image
        test_image = Image.open(test_image_path)
        result = classifier.predict(test_image)
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error processing test image: {e}")
