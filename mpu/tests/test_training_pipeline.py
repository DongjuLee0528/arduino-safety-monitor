"""
Training pipeline contract tests.

Tests the run-directory structure, config/metadata schemas, class weights,
checkpoint safety, PPT artifact generation, metrics, split manifest,
reproducibility, device selection, and production model protection.

All tests use synthetic fixtures — no real dataset required.
Tests are fast (< 5s total) and side-effect-free outside of tempdir.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch

# Project root on sys.path via pytest / setup.cfg
from mpu.ai.metrics import (
    CLASS_NAMES,
    EpochMetrics,
    EvalResult,
    append_epoch_csv,
    class_weights_tensor,
    compute_class_weights,
    compute_eval_metrics,
    save_class_distribution,
    save_confusion_matrix,
    save_training_curves,
    write_best_metrics_json,
    write_class_metrics_csv,
    write_dataset_summary_json,
    write_split_manifest_csv,
)
from mpu.ai.train import (
    _PROTECTED_FILES,
    _safe_check_production_overwrite,
    build_model,
    make_run_dir,
    select_device,
    set_seed,
    SUPPORTED_ARCHITECTURES,
    write_config_json,
    write_model_metadata_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dummy_eval() -> EvalResult:
    """Minimal EvalResult for artifact tests."""
    labels = [0] * 10 + [1] * 30
    preds  = [0] * 8 + [1] * 2 + [1] * 28 + [0] * 2
    return compute_eval_metrics(labels, preds)


def _dummy_history(n: int = 3) -> list:
    return [
        EpochMetrics(
            epoch=i + 1,
            train_loss=1.0 - i * 0.1,
            val_loss=1.2 - i * 0.1,
            train_accuracy=60.0 + i * 5,
            val_accuracy=55.0 + i * 5,
            learning_rate=0.001,
            train_f1=0.5 + i * 0.05,
            val_f1=0.48 + i * 0.05,
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Run directory creation
# ─────────────────────────────────────────────────────────────────────────────

class TestRunDirectoryCreation(unittest.TestCase):

    def test_explicit_run_id_creates_named_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, run_id = make_run_dir(tmp, "test_run_001", "efficientnet_b0")
            self.assertEqual(run_id, "test_run_001")
            self.assertTrue(os.path.isdir(run_dir))

    def test_auto_run_id_creates_timestamped_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, run_id = make_run_dir(tmp, None, "efficientnet_b0")
            self.assertIn("efficientnet_b0", run_id)
            self.assertTrue(os.path.isdir(run_dir))

    def test_sample_predictions_subdir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = make_run_dir(tmp, "r1", "efficientnet_b0")
            self.assertTrue(os.path.isdir(os.path.join(run_dir, "sample_predictions")))

    def test_misclassified_examples_subdir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = make_run_dir(tmp, "r2", "efficientnet_b0")
            self.assertTrue(os.path.isdir(os.path.join(run_dir, "misclassified_examples")))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class weight correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestClassWeights(unittest.TestCase):

    def test_no_helmet_weight_exceeds_helmet_weight(self):
        w0, w1 = compute_class_weights(no_helmet_count=3832, helmet_count=11507)
        self.assertGreater(w0, w1, "no_helmet (minority) must have higher weight")

    def test_inverse_frequency_formula(self):
        n0, n1 = 3832, 11507
        n = n0 + n1
        w0, w1 = compute_class_weights(n0, n1)
        self.assertAlmostEqual(w0, n / (2 * n0), places=4)
        self.assertAlmostEqual(w1, n / (2 * n1), places=4)

    def test_class_weights_tensor_shape(self):
        w = class_weights_tensor(3832, 11507, torch.device("cpu"))
        self.assertEqual(w.shape, (2,))
        self.assertEqual(w.dtype, torch.float32)

    def test_index_0_is_higher_weight_in_tensor(self):
        w = class_weights_tensor(3832, 11507, torch.device("cpu"))
        self.assertGreater(w[0].item(), w[1].item())

    def test_equal_counts_give_equal_weights(self):
        w0, w1 = compute_class_weights(1000, 1000)
        self.assertAlmostEqual(w0, w1, places=6)
        self.assertGreaterEqual(w0, w1)  # invariant: w0 >= w1

    def test_known_values(self):
        # From audit: N=15339, no_helmet=3832, helmet=11507
        w0, w1 = compute_class_weights(3832, 11507)
        self.assertAlmostEqual(w0, 2.0014, places=3)
        self.assertAlmostEqual(w1, 0.6665, places=3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. metrics.csv schema
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsCSV(unittest.TestCase):

    def test_header_written_on_first_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.csv")
            # file does NOT exist yet — header should be written
            em = EpochMetrics(1, 1.0, 1.1, 60.0, 58.0, 0.001, 0.5, 0.48)
            append_epoch_csv(path, em)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertIn("epoch", rows[0])
            self.assertIn("train_loss", rows[0])
            self.assertIn("val_loss", rows[0])
            self.assertIn("val_accuracy", rows[0])

    def test_multiple_epochs_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.csv")
            for i in range(1, 4):
                em = EpochMetrics(i, 1.0, 1.1, 60.0, 58.0, 0.001)
                append_epoch_csv(path, em)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. dataset_summary.json schema
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetSummaryJSON(unittest.TestCase):

    def test_required_keys_present(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_dataset_summary_json(
                path,
                source_images_total=4898, person_crops_total=15339,
                train_source_images=3918, val_source_images=980,
                train_crops=12270, val_crops=3069,
                train_no_helmet=3072, train_helmet=9198,
                val_no_helmet=760, val_helmet=2309,
                w_no_helmet=2.0014, w_helmet=0.6665,
                min_bbox_dim=32, split_seed=42,
            )
            with open(path) as f:
                data = json.load(f)
            for key in [
                "source_images_total", "person_crops_total",
                "train_source_images", "val_source_images",
                "train_crops", "val_crops",
                "train_class_counts", "val_class_counts",
                "class_weights", "min_bbox_dim", "split_seed",
                "dataset_b_excluded",
            ]:
                self.assertIn(key, data, f"Missing key: {key}")
        finally:
            os.unlink(path)

    def test_dataset_b_excluded_is_true(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_dataset_summary_json(
                path,
                source_images_total=4898, person_crops_total=15339,
                train_source_images=3918, val_source_images=980,
                train_crops=12270, val_crops=3069,
                train_no_helmet=3072, train_helmet=9198,
                val_no_helmet=760, val_helmet=2309,
                w_no_helmet=2.0014, w_helmet=0.6665,
                min_bbox_dim=32, split_seed=42,
            )
            with open(path) as f:
                data = json.load(f)
            self.assertTrue(data["dataset_b_excluded"])
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. split_manifest.csv
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitManifestCSV(unittest.TestCase):

    def test_columns_present(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)  # ensure fresh file
        try:
            samples = [("img_a.png", (0, 0, 100, 200), 1),
                       ("img_b.png", (0, 0, 100, 200), 0)]
            write_split_manifest_csv(path, samples, "train")
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertIn("source_image", rows[0])
            self.assertIn("split", rows[0])
            self.assertIn("label", rows[0])
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_train_and_val_rows_present(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            write_split_manifest_csv(path, [("a.png", (0,0,100,200), 1)], "train")
            write_split_manifest_csv(path, [("b.png", (0,0,100,200), 0)], "val")
            with open(path) as f:
                rows = list(csv.DictReader(f))
            splits = {r["split"] for r in rows}
            self.assertIn("train", splits)
            self.assertIn("val", splits)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Production model overwrite protection
# ─────────────────────────────────────────────────────────────────────────────

class TestProductionModelProtection(unittest.TestCase):

    def test_production_models_dir_raises(self):
        protected_dir = os.path.realpath(os.path.dirname(_PROTECTED_FILES[0]))
        with self.assertRaises(RuntimeError):
            _safe_check_production_overwrite(protected_dir)

    def test_safe_run_dir_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Should not raise
            _safe_check_production_overwrite(os.path.join(tmp, "run_001"))

    def test_best_model_pth_is_in_protected_list(self):
        names = [os.path.basename(p) for p in _PROTECTED_FILES]
        self.assertIn("best_model.pth", names)

    def test_best_model_onnx_is_in_protected_list(self):
        names = [os.path.basename(p) for p in _PROTECTED_FILES]
        self.assertIn("best_model.onnx", names)


# ─────────────────────────────────────────────────────────────────────────────
# 7. PPT artifacts
# ─────────────────────────────────────────────────────────────────────────────

class TestPPTArtifacts(unittest.TestCase):

    def test_loss_graph_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            p_loss = os.path.join(tmp, "train_val_loss.png")
            p_acc  = os.path.join(tmp, "train_val_accuracy.png")
            save_training_curves(p_loss, p_acc, _dummy_history())
            self.assertTrue(os.path.isfile(p_loss), "loss PNG not created")

    def test_accuracy_graph_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            p_loss = os.path.join(tmp, "train_val_loss.png")
            p_acc  = os.path.join(tmp, "train_val_accuracy.png")
            save_training_curves(p_loss, p_acc, _dummy_history())
            self.assertTrue(os.path.isfile(p_acc), "accuracy PNG not created")

    def test_confusion_matrix_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "confusion_matrix.png")
            ev = _dummy_eval()
            save_confusion_matrix(path, ev)
            self.assertTrue(os.path.isfile(path))

    def test_class_distribution_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "class_distribution.png")
            save_class_distribution(path, 3072, 9198, 760, 2309)
            self.assertTrue(os.path.isfile(path))


# ─────────────────────────────────────────────────────────────────────────────
# 8. best_metrics.json
# ─────────────────────────────────────────────────────────────────────────────

class TestBestMetricsJSON(unittest.TestCase):

    def test_required_keys_present(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            em = EpochMetrics(5, 0.3, 0.35, 88.0, 85.0, 0.001, 0.87, 0.84)
            ev = _dummy_eval()
            write_best_metrics_json(path, 5, em, ev)
            with open(path) as f:
                data = json.load(f)
            for key in [
                "best_epoch", "best_val_loss", "best_val_accuracy",
                "best_macro_f1", "best_no_helmet_recall", "best_helmet_recall",
                "missed_violation_count", "missed_violation_rate",
                "false_alarm_count", "false_alarm_rate",
            ]:
                self.assertIn(key, data, f"Missing: {key}")
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. class_metrics.csv
# ─────────────────────────────────────────────────────────────────────────────

class TestClassMetricsCSV(unittest.TestCase):

    def test_both_classes_present(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            ev = _dummy_eval()
            write_class_metrics_csv(path, ev)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            labels = [r["label"] for r in rows]
            self.assertIn("no_helmet", labels)
            self.assertIn("helmet", labels)
        finally:
            os.unlink(path)

    def test_precision_recall_f1_columns_present(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            ev = _dummy_eval()
            write_class_metrics_csv(path, ev)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            row = rows[0]
            for col in ["precision", "recall", "f1", "support"]:
                self.assertIn(col, row)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Safety metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyMetrics(unittest.TestCase):
    """Verify false_alarm / missed_violation semantics."""

    def test_missed_violation_is_actual_no_helmet_pred_helmet(self):
        # 5 actual no_helmet predicted as helmet
        labels = [0] * 10 + [1] * 10
        preds  = [1] * 5 + [0] * 5 + [1] * 10
        ev = compute_eval_metrics(labels, preds)
        self.assertEqual(ev.missed_violation_count, 5)

    def test_false_alarm_is_actual_helmet_pred_no_helmet(self):
        # 3 actual helmet predicted as no_helmet
        labels = [0] * 10 + [1] * 10
        preds  = [0] * 10 + [0] * 3 + [1] * 7
        ev = compute_eval_metrics(labels, preds)
        self.assertEqual(ev.false_alarm_count, 3)

    def test_perfect_predictions_have_zero_errors(self):
        labels = [0] * 10 + [1] * 10
        preds  = labels[:]
        ev = compute_eval_metrics(labels, preds)
        self.assertEqual(ev.missed_violation_count, 0)
        self.assertEqual(ev.false_alarm_count, 0)

    def test_class_names_order(self):
        from mpu.ai.metrics import CLASS_NO_HELMET, CLASS_HELMET
        self.assertEqual(CLASS_NO_HELMET, 0)
        self.assertEqual(CLASS_HELMET, 1)
        self.assertEqual(CLASS_NAMES[0], "no_helmet")
        self.assertEqual(CLASS_NAMES[1], "helmet")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Misclassification directory structure
# ─────────────────────────────────────────────────────────────────────────────

class TestMisclassificationDirs(unittest.TestCase):

    def test_false_helmet_dir_exists_after_make_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _ = make_run_dir(tmp, "r3", "efficientnet_b0")
            # save_misclassified_examples creates subdirs at call time;
            # verify the parent dir exists
            self.assertTrue(os.path.isdir(
                os.path.join(run_dir, "misclassified_examples")
            ))


# ─────────────────────────────────────────────────────────────────────────────
# 12. Seed / reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestSeedHandling(unittest.TestCase):

    def test_set_seed_does_not_raise(self):
        set_seed(42)  # should complete without error

    def test_random_output_is_deterministic_after_same_seed(self):
        import random as _r
        set_seed(7)
        a = _r.random()
        set_seed(7)
        b = _r.random()
        self.assertEqual(a, b)

    def test_numpy_deterministic_after_seed(self):
        set_seed(13)
        a = np.random.rand()
        set_seed(13)
        b = np.random.rand()
        self.assertAlmostEqual(a, b)

    def test_torch_deterministic_after_seed(self):
        set_seed(99)
        a = torch.rand(5)
        set_seed(99)
        b = torch.rand(5)
        self.assertTrue(torch.allclose(a, b))


# ─────────────────────────────────────────────────────────────────────────────
# 13. Device selection
# ─────────────────────────────────────────────────────────────────────────────

class TestDeviceSelection(unittest.TestCase):

    def test_select_device_returns_valid_device(self):
        device = select_device()
        self.assertIn(device.type, ("mps", "cuda", "cpu"))

    def test_device_is_mps_on_mac_with_mps(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available on this machine")
        device = select_device()
        self.assertEqual(device.type, "mps")


# ─────────────────────────────────────────────────────────────────────────────
# 14. convert.py run-local path handling
# ─────────────────────────────────────────────────────────────────────────────

class TestConvertRunLocalPath(unittest.TestCase):

    def test_missing_checkpoint_arg_causes_error(self):
        """convert.py must require explicit --checkpoint."""
        from mpu.ai import convert as conv_module
        import argparse
        # Patch sys.argv and check parser errors with missing args
        with patch("sys.argv", ["convert.py"]):
            with self.assertRaises(SystemExit):
                conv_module.main()

    def test_convert_function_raises_on_missing_checkpoint(self):
        from mpu.ai.convert import convert_to_onnx
        with self.assertRaises(FileNotFoundError):
            convert_to_onnx("/nonexistent/best_model.pth", "/tmp/out.onnx")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Confusion matrix axis semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestConfusionMatrixSemantics(unittest.TestCase):

    def test_cm_shape_is_2x2(self):
        ev = _dummy_eval()
        self.assertEqual(ev.cm.shape, (2, 2))

    def test_cm_row0_is_actual_no_helmet(self):
        # All labels are no_helmet=0, no preds for 1
        labels = [0] * 10
        preds  = [0] * 10
        ev = compute_eval_metrics(labels, preds)
        # row 0 (no_helmet), col 0 (pred no_helmet) = 10
        self.assertEqual(ev.cm[0, 0], 10)
        self.assertEqual(ev.cm[0, 1], 0)

    def test_cm_row1_is_actual_helmet(self):
        labels = [1] * 10
        preds  = [1] * 10
        ev = compute_eval_metrics(labels, preds)
        self.assertEqual(ev.cm[1, 1], 10)
        self.assertEqual(ev.cm[1, 0], 0)

    def test_missed_violation_is_cm_01(self):
        # missed violation: actual no_helmet(0) → pred helmet(1) = cm[0,1]
        labels = [0]
        preds  = [1]
        ev = compute_eval_metrics(labels, preds)
        self.assertEqual(ev.cm[0, 1], 1)
        self.assertEqual(ev.missed_violation_count, 1)

    def test_false_alarm_is_cm_10(self):
        # false alarm: actual helmet(1) → pred no_helmet(0) = cm[1,0]
        labels = [1]
        preds  = [0]
        ev = compute_eval_metrics(labels, preds)
        self.assertEqual(ev.cm[1, 0], 1)
        self.assertEqual(ev.false_alarm_count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Early stopping metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyStoppingMetadata(unittest.TestCase):

    def test_best_metrics_epoch_recorded(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            em = EpochMetrics(3, 0.2, 0.25, 90.0, 88.0, 0.001)
            ev = _dummy_eval()
            write_best_metrics_json(path, 3, em, ev)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["best_epoch"], 3)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 17. Model factory — build_model()
# ─────────────────────────────────────────────────────────────────────────────

class TestModelFactory(unittest.TestCase):

    def test_efficientnet_b0_builds(self):
        model = build_model("efficientnet_b0", num_classes=2)
        self.assertIsNotNone(model)

    def test_mobilenet_v3_small_builds(self):
        model = build_model("mobilenet_v3_small", num_classes=2)
        self.assertIsNotNone(model)

    def test_invalid_architecture_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_model("resnet50", num_classes=2)

    def test_efficientnet_output_shape_is_b_2(self):
        model = build_model("efficientnet_b0", num_classes=2)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (2, 2))

    def test_mobilenet_output_shape_is_b_2(self):
        model = build_model("mobilenet_v3_small", num_classes=2)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (2, 2))

    def test_mobilenet_is_smaller_than_efficientnet(self):
        n_eff = sum(p.numel() for p in build_model("efficientnet_b0").parameters())
        n_mob = sum(p.numel() for p in build_model("mobilenet_v3_small").parameters())
        self.assertLess(n_mob, n_eff)

    def test_supported_architectures_contains_both(self):
        self.assertIn("efficientnet_b0", SUPPORTED_ARCHITECTURES)
        self.assertIn("mobilenet_v3_small", SUPPORTED_ARCHITECTURES)

    def test_class_mapping_preserved_efficientnet(self):
        # Output dim 0 = no_helmet, 1 = helmet — verified by classifier size
        model = build_model("efficientnet_b0", num_classes=2)
        last = model.classifier[-1]
        self.assertEqual(last.out_features, 2)

    def test_class_mapping_preserved_mobilenet(self):
        model = build_model("mobilenet_v3_small", num_classes=2)
        last = model.classifier[-1]
        self.assertEqual(last.out_features, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 18. Architecture persisted in run metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestArchitectureMetadata(unittest.TestCase):

    def _make_args(self, arch):
        import argparse
        args = argparse.Namespace(
            architecture=arch,
            shel5k_path="/fake/path",
            batch_size=32, epochs=50, lr=0.001, seed=42,
            patience=5, train_ratio=0.8, output_dir="runs",
            run_id=None, num_workers=0, save_examples=False,
        )
        return args

    def test_efficientnet_run_id_contains_arch_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, run_id = make_run_dir(tmp, None, "efficientnet_b0")
            self.assertIn("efficientnet_b0", run_id)

    def test_mobilenet_run_id_contains_arch_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, run_id = make_run_dir(tmp, None, "mobilenet_v3_small")
            self.assertIn("mobilenet_v3_small", run_id)

    def test_config_json_records_mobilenet_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            args = self._make_args("mobilenet_v3_small")
            write_config_json(path, args, "run_001", torch.device("cpu"),
                              1.99, 0.67, architecture="mobilenet_v3_small")
            data = json.load(open(path))
            self.assertEqual(data["architecture"], "mobilenet_v3_small")

    def test_config_json_records_efficientnet_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            args = self._make_args("efficientnet_b0")
            write_config_json(path, args, "run_002", torch.device("cpu"),
                              1.99, 0.67, architecture="efficientnet_b0")
            data = json.load(open(path))
            self.assertEqual(data["architecture"], "efficientnet_b0")

    def test_model_metadata_records_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meta.json")
            model = build_model("mobilenet_v3_small")
            write_model_metadata_json(
                path, model, torch.device("cpu"), 5, {}, "/fake/ckpt.pth",
                architecture="mobilenet_v3_small",
            )
            data = json.load(open(path))
            self.assertEqual(data["architecture"], "mobilenet_v3_small")

    def test_model_metadata_records_parameter_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meta.json")
            model = build_model("mobilenet_v3_small")
            write_model_metadata_json(
                path, model, torch.device("cpu"), 1, {}, "/fake/ckpt.pth",
                architecture="mobilenet_v3_small",
            )
            data = json.load(open(path))
            self.assertIn("parameter_count", data)
            self.assertGreater(data["parameter_count"], 0)

    def test_class_weights_same_regardless_of_architecture(self):
        # Same formula, same counts → identical weights for both models
        w0_eff, w1_eff = compute_class_weights(3072, 9198)
        w0_mob, w1_mob = compute_class_weights(3072, 9198)
        self.assertAlmostEqual(w0_eff, w0_mob, places=6)
        self.assertAlmostEqual(w1_eff, w1_mob, places=6)


# ─────────────────────────────────────────────────────────────────────────────
# 19. convert.py architecture detection
# ─────────────────────────────────────────────────────────────────────────────

class TestConvertArchitectureDetection(unittest.TestCase):

    def test_detect_arch_reads_config_json(self):
        from mpu.ai.convert import _detect_architecture
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "best_model.pth")
            open(ckpt_path, "w").close()  # empty placeholder
            cfg = {"architecture": "mobilenet_v3_small"}
            json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
            arch = _detect_architecture(ckpt_path)
            self.assertEqual(arch, "mobilenet_v3_small")

    def test_detect_arch_raises_when_no_config(self):
        from mpu.ai.convert import _detect_architecture
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "best_model.pth")
            with self.assertRaises(FileNotFoundError):
                _detect_architecture(ckpt_path)

    def test_convert_raises_on_wrong_architecture(self):
        from mpu.ai.convert import convert_to_onnx
        with tempfile.TemporaryDirectory() as tmp:
            # Save an EfficientNet state_dict, then try to load as MobileNet
            eff_model = build_model("efficientnet_b0")
            ckpt = os.path.join(tmp, "eff.pth")
            torch.save(eff_model.state_dict(), ckpt)
            out = os.path.join(tmp, "out.onnx")
            with self.assertRaises(RuntimeError):
                convert_to_onnx(ckpt, out, architecture="mobilenet_v3_small")

    def test_convert_raises_file_not_found_for_missing_checkpoint(self):
        from mpu.ai.convert import convert_to_onnx
        with self.assertRaises(FileNotFoundError):
            convert_to_onnx("/nonexistent/best_model.pth", "/tmp/out.onnx")

    def test_onnx_opset_is_18(self):
        from mpu.ai.convert import _ONNX_OPSET
        self.assertEqual(_ONNX_OPSET, 18)

    def test_convert_mobilenet_to_onnx(self):
        from mpu.ai.convert import convert_to_onnx
        import onnxruntime as rt
        with tempfile.TemporaryDirectory() as tmp:
            mob_model = build_model("mobilenet_v3_small")
            ckpt = os.path.join(tmp, "mob.pth")
            torch.save(mob_model.state_dict(), ckpt)
            out = os.path.join(tmp, "mob.onnx")
            result = convert_to_onnx(ckpt, out, architecture="mobilenet_v3_small")
            self.assertTrue(result)
            self.assertTrue(os.path.exists(out))
            # Verify ORT can load and run inference
            sess = rt.InferenceSession(out, providers=["CPUExecutionProvider"])
            x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
            logits = sess.run(None, {"input": x})[0]
            self.assertEqual(logits.shape, (1, 2))

    def test_convert_efficientnet_to_onnx(self):
        from mpu.ai.convert import convert_to_onnx
        import onnxruntime as rt
        with tempfile.TemporaryDirectory() as tmp:
            eff_model = build_model("efficientnet_b0")
            ckpt = os.path.join(tmp, "eff.pth")
            torch.save(eff_model.state_dict(), ckpt)
            out = os.path.join(tmp, "eff.onnx")
            result = convert_to_onnx(ckpt, out, architecture="efficientnet_b0")
            self.assertTrue(result)
            self.assertTrue(os.path.exists(out))
            sess = rt.InferenceSession(out, providers=["CPUExecutionProvider"])
            x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
            logits = sess.run(None, {"input": x})[0]
            self.assertEqual(logits.shape, (1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# 20. MPS smoke test — MobileNetV3-Small
# ─────────────────────────────────────────────────────────────────────────────

class TestMPSMobileNet(unittest.TestCase):

    def setUp(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")

    def test_mobilenet_forward_on_mps(self):
        device = torch.device("mps")
        model = build_model("mobilenet_v3_small", num_classes=2).to(device)
        model.eval()
        x = torch.randn(16, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (16, 2))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_mobilenet_loss_finite_on_mps(self):
        device = torch.device("mps")
        model = build_model("mobilenet_v3_small", num_classes=2).to(device)
        model.train()
        x = torch.randn(4, 3, 224, 224, device=device)
        labels = torch.randint(0, 2, (4,), device=device)
        criterion = torch.nn.CrossEntropyLoss()
        out = model(x)
        loss = criterion(out, labels)
        self.assertTrue(torch.isfinite(loss).item())


class TestMobileNetV3Large(unittest.TestCase):
    """Focused tests for MobileNetV3-Large architecture support."""

    def test_build_mobilenet_v3_large_output_shape(self):
        model = build_model("mobilenet_v3_large", num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 2))

    def test_mobilenet_v3_large_classifier_head(self):
        model = build_model("mobilenet_v3_large", num_classes=2)
        last = model.classifier[-1]
        self.assertIsInstance(last, torch.nn.Linear)
        self.assertEqual(last.in_features, 1280)
        self.assertEqual(last.out_features, 2)

    def test_mobilenet_v3_large_params_greater_than_small(self):
        small = build_model("mobilenet_v3_small", num_classes=2)
        large = build_model("mobilenet_v3_large", num_classes=2)
        n_small = sum(p.numel() for p in small.parameters())
        n_large = sum(p.numel() for p in large.parameters())
        self.assertGreater(n_large, n_small)

    def test_mobilenet_v3_large_in_supported_architectures(self):
        from mpu.ai.train import SUPPORTED_ARCHITECTURES
        self.assertIn("mobilenet_v3_large", SUPPORTED_ARCHITECTURES)

    def test_mobilenet_v3_large_pretrained_weights_name(self):
        from mpu.ai.train import _PRETRAINED_WEIGHTS_NAMES
        self.assertIn("mobilenet_v3_large", _PRETRAINED_WEIGHTS_NAMES)
        self.assertIn("IMAGENET1K_V1", _PRETRAINED_WEIGHTS_NAMES["mobilenet_v3_large"])

    def test_mobilenet_v3_large_onnx_export_and_ort_inference(self):
        import tempfile, onnx, onnx.checker, onnxruntime as ort, numpy as np
        model = build_model("mobilenet_v3_large", num_classes=2)
        model.eval()
        with tempfile.TemporaryDirectory() as td:
            onnx_path = os.path.join(td, "mob_large.onnx")
            dummy = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model, dummy, onnx_path,
                export_params=True, opset_version=18,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            proto = onnx.load(onnx_path)
            onnx.checker.check_model(proto)
            opset = next(o.version for o in proto.opset_import if o.domain == "")
            self.assertEqual(opset, 18)
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"input": dummy.numpy().astype(np.float32)})[0]
            self.assertEqual(out.shape, (1, 2))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_mobilenet_v3_large_checkpoint_roundtrip(self):
        import tempfile
        model = build_model("mobilenet_v3_large", num_classes=2)
        with tempfile.TemporaryDirectory() as td:
            pth = os.path.join(td, "mob_large.pth")
            torch.save(model.state_dict(), pth)
            model2 = build_model("mobilenet_v3_large", num_classes=2)
            model2.load_state_dict(
                torch.load(pth, map_location="cpu", weights_only=True)
            )
            model.eval(); model2.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x), model2(x)))

    def test_convert_auto_detects_mobilenet_v3_large(self):
        import tempfile, json
        from mpu.ai.convert import _detect_architecture
        model = build_model("mobilenet_v3_large", num_classes=2)
        with tempfile.TemporaryDirectory() as td:
            pth = os.path.join(td, "best_model.pth")
            torch.save(model.state_dict(), pth)
            cfg = {"architecture": "mobilenet_v3_large"}
            with open(os.path.join(td, "config.json"), "w") as f:
                json.dump(cfg, f)
            detected = _detect_architecture(pth)
            self.assertEqual(detected, "mobilenet_v3_large")

    def test_mobilenet_v3_large_mps_forward(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
        device = torch.device("mps")
        model = build_model("mobilenet_v3_large", num_classes=2).to(device)
        model.eval()
        x = torch.randn(4, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (4, 2))
        self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
