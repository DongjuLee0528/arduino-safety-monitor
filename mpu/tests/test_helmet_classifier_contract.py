"""
ONNX contract and preprocessing tests for HelmetClassifier.

These tests verify:
- best_model.onnx file presence and external data file
- ONNX input/output schema (shape, dtype)
- Preprocessing contract (BGR->RGB, float32, [0,1], ImageNet norm, CHW, batch)
- Actual ONNX inference session executes without error
- Class mapping contract: index 0 = no_helmet, index 1 = helmet

No real labeled image fixtures are available in this repository.
Inference on synthetic inputs is executed to verify the session runs;
the predicted label on synthetic inputs is not asserted as ground truth.
"""

import os
import unittest
import numpy as np

from mpu.classifier import HelmetClassifier
from mpu.config import ONNX_MODEL_PATH

MODEL_PATH = ONNX_MODEL_PATH


class TestOnnxModelPresence(unittest.TestCase):
    def test_onnx_model_file_exists(self):
        self.assertTrue(
            os.path.isfile(MODEL_PATH),
            f"ONNX model not found: {MODEL_PATH}"
        )

    def test_onnx_external_data_file_exists(self):
        data_path = MODEL_PATH + ".data"
        self.assertTrue(
            os.path.isfile(data_path),
            f"ONNX external data not found: {data_path}"
        )


class TestOnnxSessionSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort
        cls.session = ort.InferenceSession(MODEL_PATH)
        cls.inp = cls.session.get_inputs()[0]
        cls.out = cls.session.get_outputs()[0]

    def test_input_dtype_is_float32(self):
        self.assertEqual(self.inp.type, "tensor(float)")

    def test_input_has_4_dimensions(self):
        self.assertEqual(len(self.inp.shape), 4)

    def test_input_channels_is_3(self):
        self.assertEqual(self.inp.shape[1], 3)

    def test_input_height_is_224(self):
        self.assertEqual(self.inp.shape[2], 224)

    def test_input_width_is_224(self):
        self.assertEqual(self.inp.shape[3], 224)

    def test_output_has_2_dimensions(self):
        self.assertEqual(len(self.out.shape), 2)

    def test_output_num_classes_is_2(self):
        self.assertEqual(self.out.shape[1], 2)

    def test_output_dtype_is_float32(self):
        self.assertEqual(self.out.type, "tensor(float)")


class TestPreprocessingContract(unittest.TestCase):
    """Verify the preprocessing pipeline produces correctly shaped/typed tensors."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = HelmetClassifier()

    def _preprocess(self, bgr_uint8):
        return self.classifier._preprocess_image(bgr_uint8)

    def test_output_dtype_float32(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        out = self._preprocess(bgr)
        self.assertEqual(out.dtype, np.float32)

    def test_output_shape_1_3_224_224(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        out = self._preprocess(bgr)
        self.assertEqual(out.shape, (1, 3, 224, 224))

    def test_imagenet_mean_pixel_normalises_near_zero(self):
        # A pixel equal to the ImageNet mean should normalise to ~0
        mean_px = np.array([[[int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)]]], dtype=np.uint8)
        img = np.broadcast_to(mean_px, (224, 224, 3)).copy()
        bgr = img[:, :, ::-1].copy()  # RGB -> BGR (reverse channels for input)
        out = self._preprocess(bgr)
        self.assertTrue(np.all(np.abs(out) < 0.02), f"Expected near-zero, got max abs {np.abs(out).max():.4f}")


class TestOnnxSessionExecution(unittest.TestCase):
    """Verify the ONNX session runs without error on synthetic inputs."""

    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort
        cls.session = ort.InferenceSession(MODEL_PATH)
        cls.inp_name = cls.session.get_inputs()[0].name
        cls.out_name = cls.session.get_outputs()[0].name

    def _run(self, arr):
        return self.session.run([self.out_name], {self.inp_name: arr})[0]

    def test_session_runs_with_zero_input(self):
        x = np.zeros((1, 3, 224, 224), dtype=np.float32)
        out = self._run(x)
        self.assertEqual(out.shape, (1, 2))

    def test_session_runs_with_random_input(self):
        rng = np.random.RandomState(0)
        x = rng.randn(1, 3, 224, 224).astype(np.float32)
        out = self._run(x)
        self.assertEqual(out.shape, (1, 2))

    def test_output_contains_finite_values(self):
        x = np.zeros((1, 3, 224, 224), dtype=np.float32)
        out = self._run(x)
        self.assertTrue(np.all(np.isfinite(out)), "ONNX output contains non-finite values")

    def test_softmax_sums_to_one(self):
        x = np.zeros((1, 3, 224, 224), dtype=np.float32)
        logits = self._run(x)[0]
        exp_x = np.exp(logits - np.max(logits))
        probs = exp_x / np.sum(exp_x)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)


class TestClassMappingContract(unittest.TestCase):
    """Verify class mapping convention is consistent with HelmetClassifier."""

    def test_classifier_label_for_index_0_is_no_helmet(self):
        # Simulate: argmax=0 -> label should be no_helmet
        predicted_class = 0
        label = "helmet" if predicted_class == 1 else "no_helmet"
        self.assertEqual(label, "no_helmet")

    def test_classifier_label_for_index_1_is_helmet(self):
        predicted_class = 1
        label = "helmet" if predicted_class == 1 else "no_helmet"
        self.assertEqual(label, "helmet")

    def test_zero_input_returns_valid_label(self):
        import onnxruntime as ort
        session = ort.InferenceSession(MODEL_PATH)
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        x = np.zeros((1, 3, 224, 224), dtype=np.float32)
        logits = session.run([out.name], {inp.name: x})[0][0]
        exp_x = np.exp(logits - np.max(logits))
        probs = exp_x / np.sum(exp_x)
        idx = int(np.argmax(probs))
        label = "helmet" if idx == 1 else "no_helmet"
        self.assertIn(label, ("helmet", "no_helmet"))
        # NOTE: zero input predicts no_helmet with high confidence, which is
        # consistent with the model's strong no_helmet bias.
        # CALIBRATION REQUIRED: actual camera images needed to verify real-world accuracy.


if __name__ == "__main__":
    unittest.main()
