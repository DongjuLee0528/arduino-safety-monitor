"""
Tests for the daily statistics layer:
  - DashboardState.update_statistics() validation and rollover
  - HelmetDetectionSystem per-frame counting via IoU-based new-worker detection
  - No regression of the detection pipeline
"""

import unittest
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np

from mpu.dashboard_state import DashboardState, HelmetResult, STATISTICS_TIMEZONE
from mpu.main import HelmetDetectionSystem, _bbox_iou, _is_new_worker, _EventSuppressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KST = ZoneInfo("Asia/Seoul")


def _kst_today():
    return datetime.now(_KST).date()


def _utc_today():
    return datetime.now(timezone.utc).date()


def _make_system():
    system = HelmetDetectionSystem.__new__(HelmetDetectionSystem)
    system.camera = MagicMock()
    system.person_detector = MagicMock()
    system.helmet_classifier = MagicMock()
    system.sender = MagicMock()
    system.bridge_rpc = MagicMock()
    system.alert_manager = MagicMock()
    system.running = False
    system.alert_hardware_active = False
    system._connected = False
    system.dashboard = DashboardState()
    system._events = _EventSuppressor(system.dashboard, suppress_seconds=0.0)
    system._prev_bboxes = []
    return system


def _frame():
    return np.zeros((480, 640, 3), dtype="uint8")


def _person(bbox, label="helmet", confidence=0.9):
    return {"bbox": bbox, "label": label, "confidence": confidence}


def _snap_stats(system):
    return system.dashboard.snapshot()["statistics"]


# ---------------------------------------------------------------------------
# Unit tests for _bbox_iou
# ---------------------------------------------------------------------------

class TestBboxIou(unittest.TestCase):
    def test_identical_boxes_iou_is_one(self):
        self.assertAlmostEqual(_bbox_iou([0, 0, 100, 100], [0, 0, 100, 100]), 1.0)

    def test_no_overlap_iou_is_zero(self):
        self.assertAlmostEqual(_bbox_iou([0, 0, 50, 50], [100, 100, 50, 50]), 0.0)

    def test_partial_overlap(self):
        iou = _bbox_iou([0, 0, 100, 100], [50, 50, 100, 100])
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_zero_area_box_returns_zero(self):
        self.assertEqual(_bbox_iou([0, 0, 0, 100], [0, 0, 100, 100]), 0.0)

    def test_adjacent_boxes_no_overlap(self):
        self.assertAlmostEqual(_bbox_iou([0, 0, 50, 50], [50, 0, 50, 50]), 0.0)


# ---------------------------------------------------------------------------
# Unit tests for _is_new_worker
# ---------------------------------------------------------------------------

class TestIsNewWorker(unittest.TestCase):
    def test_empty_prev_is_always_new(self):
        self.assertTrue(_is_new_worker([0, 0, 100, 100], []))

    def test_same_bbox_is_not_new(self):
        self.assertFalse(_is_new_worker([0, 0, 100, 100], [[0, 0, 100, 100]]))

    def test_clearly_different_bbox_is_new(self):
        self.assertTrue(_is_new_worker([400, 300, 100, 100], [[0, 0, 100, 100]]))

    def test_slight_jitter_is_not_new(self):
        self.assertFalse(_is_new_worker([2, 2, 98, 98], [[0, 0, 100, 100]]))

    def test_matches_one_of_multiple_prev(self):
        prev = [[0, 0, 100, 100], [300, 300, 100, 100]]
        self.assertFalse(_is_new_worker([0, 0, 100, 100], prev))


# ---------------------------------------------------------------------------
# Unit tests for DashboardState.update_statistics
# ---------------------------------------------------------------------------

class TestUpdateStatistics(unittest.TestCase):
    def test_increments_inspected(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=3)
        self.assertEqual(ds.snapshot()["statistics"]["inspected"], 3)

    def test_increments_helmet(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=1, helmet_delta=1)
        self.assertEqual(ds.snapshot()["statistics"]["helmet"], 1)

    def test_increments_no_helmet(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=1, no_helmet_delta=1)
        self.assertEqual(ds.snapshot()["statistics"]["no_helmet"], 1)

    def test_cumulative_increments(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=2, helmet_delta=1, no_helmet_delta=1)
        ds.update_statistics(inspected_delta=1, helmet_delta=1)
        snap = ds.snapshot()["statistics"]
        self.assertEqual(snap["inspected"], 3)
        self.assertEqual(snap["helmet"], 2)
        self.assertEqual(snap["no_helmet"], 1)

    def test_zero_delta_no_change(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=5)
        ds.update_statistics()
        self.assertEqual(ds.snapshot()["statistics"]["inspected"], 5)

    def test_date_rollover_resets_counters(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=10, helmet_delta=7, no_helmet_delta=3)
        yesterday = _kst_today() - timedelta(days=1)
        with ds._lock:
            ds._stats_date = yesterday
        ds.update_statistics(inspected_delta=1, helmet_delta=1)
        snap = ds.snapshot()["statistics"]
        self.assertEqual(snap["inspected"], 1)
        self.assertEqual(snap["helmet"], 1)
        self.assertEqual(snap["no_helmet"], 0)

    def test_date_after_rollover_is_today_kst(self):
        ds = DashboardState()
        yesterday = _kst_today() - timedelta(days=1)
        with ds._lock:
            ds._stats_date = yesterday
        ds.update_statistics()
        self.assertEqual(ds.snapshot()["statistics"]["date"], _kst_today().isoformat())

    def test_negative_inspected_raises(self):
        ds = DashboardState()
        with self.assertRaises(ValueError):
            ds.update_statistics(inspected_delta=-1)

    def test_negative_helmet_raises(self):
        ds = DashboardState()
        with self.assertRaises(ValueError):
            ds.update_statistics(helmet_delta=-1)

    def test_negative_no_helmet_raises(self):
        ds = DashboardState()
        with self.assertRaises(ValueError):
            ds.update_statistics(no_helmet_delta=-1)

    def test_float_delta_raises(self):
        ds = DashboardState()
        with self.assertRaises(TypeError):
            ds.update_statistics(inspected_delta=1.0)

    def test_bool_delta_raises(self):
        ds = DashboardState()
        with self.assertRaises(TypeError):
            ds.update_statistics(inspected_delta=True)

    def test_snapshot_statistics_keys_present(self):
        ds = DashboardState()
        snap = ds.snapshot()["statistics"]
        self.assertIn("date", snap)
        self.assertIn("inspected", snap)
        self.assertIn("helmet", snap)
        self.assertIn("no_helmet", snap)
        self.assertIn("warnings", snap)

    def test_warnings_initializes_to_zero(self):
        ds = DashboardState()
        self.assertEqual(ds.snapshot()["statistics"]["warnings"], 0)

    def test_warnings_delta_increments(self):
        ds = DashboardState()
        ds.update_statistics(warnings_delta=3)
        self.assertEqual(ds.snapshot()["statistics"]["warnings"], 3)

    def test_warnings_cumulative(self):
        ds = DashboardState()
        ds.update_statistics(warnings_delta=1)
        ds.update_statistics(warnings_delta=2)
        self.assertEqual(ds.snapshot()["statistics"]["warnings"], 3)

    def test_warnings_reset_on_rollover(self):
        ds = DashboardState()
        ds.update_statistics(warnings_delta=5)
        yesterday = _kst_today() - timedelta(days=1)
        with ds._lock:
            ds._stats_date = yesterday
        ds.update_statistics()
        self.assertEqual(ds.snapshot()["statistics"]["warnings"], 0)

    def test_negative_warnings_delta_raises(self):
        ds = DashboardState()
        with self.assertRaises(ValueError):
            ds.update_statistics(warnings_delta=-1)

    def test_bool_warnings_delta_raises(self):
        ds = DashboardState()
        with self.assertRaises(TypeError):
            ds.update_statistics(warnings_delta=True)

    def test_existing_callers_without_warnings_still_valid(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=1, helmet_delta=1, no_helmet_delta=0)
        snap = ds.snapshot()["statistics"]
        self.assertEqual(snap["warnings"], 0)
        self.assertEqual(snap["inspected"], 1)


# ---------------------------------------------------------------------------
# Integration tests: HelmetDetectionSystem counting behaviour
# ---------------------------------------------------------------------------

class TestWorkerCounting(unittest.TestCase):
    def test_single_worker_counted_once_on_first_frame(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}

        system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 1)

    def test_same_worker_across_many_frames_counted_once(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}

        for _ in range(120):
            system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 1)

    def test_helmet_worker_increments_helmet_count(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}

        system.process_frame(_frame())

        snap = _snap_stats(system)
        self.assertEqual(snap["helmet"], 1)
        self.assertEqual(snap["no_helmet"], 0)

    def test_no_helmet_worker_increments_no_helmet_count(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.88}

        system.process_frame(_frame())

        snap = _snap_stats(system)
        self.assertEqual(snap["no_helmet"], 1)
        self.assertEqual(snap["helmet"], 0)

    def test_two_different_workers_counted_separately(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([400, 0, 100, 200]),
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": "helmet", "confidence": 0.9},
            {"label": "no_helmet", "confidence": 0.85},
        ]

        system.process_frame(_frame())

        snap = _snap_stats(system)
        self.assertEqual(snap["inspected"], 2)
        self.assertEqual(snap["helmet"], 1)
        self.assertEqual(snap["no_helmet"], 1)

    def test_two_workers_same_across_many_frames_counted_once_each(self):
        system = _make_system()
        for _ in range(60):
            system.person_detector.detect.return_value = [
                _person([0, 0, 100, 200]),
                _person([400, 0, 100, 200]),
            ]
            system.helmet_classifier.predict.side_effect = [
                {"label": "helmet", "confidence": 0.9},
                {"label": "helmet", "confidence": 0.9},
            ]
            system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 2)

    def test_worker_leaves_and_reenters_counted_twice(self):
        system = _make_system()
        bbox = [0, 0, 100, 200]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}

        system.person_detector.detect.return_value = [_person(bbox)]
        system.process_frame(_frame())

        system.person_detector.detect.return_value = []
        system.process_frame(_frame())

        system.person_detector.detect.return_value = [_person(bbox)]
        system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 2)

    def test_no_persons_stats_unchanged(self):
        system = _make_system()
        system.person_detector.detect.return_value = []

        system.process_frame(_frame())

        snap = _snap_stats(system)
        self.assertEqual(snap["inspected"], 0)
        self.assertEqual(snap["helmet"], 0)
        self.assertEqual(snap["no_helmet"], 0)

    def test_empty_crop_not_counted(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 0, 0])]

        system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 0)

    def test_statistics_snapshot_correct_after_multiple_entries(self):
        system = _make_system()
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}

        for i in range(3):
            system.person_detector.detect.return_value = [_person([i * 200, 0, 100, 200])]
            system.process_frame(_frame())
            system.person_detector.detect.return_value = []
            system.process_frame(_frame())

        snap = _snap_stats(system)
        self.assertEqual(snap["inspected"], 3)
        self.assertEqual(snap["helmet"], 3)
        self.assertEqual(snap["no_helmet"], 0)


# ---------------------------------------------------------------------------
# Date rollover integration test
# ---------------------------------------------------------------------------

class TestDateRolloverIntegration(unittest.TestCase):
    def test_date_rollover_resets_statistics(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        self.assertEqual(_snap_stats(system)["inspected"], 1)

        yesterday = _kst_today() - timedelta(days=1)
        with system.dashboard._lock:
            system.dashboard._stats_date = yesterday

        system._prev_bboxes = []
        system.process_frame(_frame())

        snap = _snap_stats(system)
        self.assertEqual(snap["inspected"], 1)
        self.assertEqual(snap["helmet"], 1)
        self.assertEqual(snap["date"], _kst_today().isoformat())


# ---------------------------------------------------------------------------
# No regression of existing detection pipeline
# ---------------------------------------------------------------------------

class TestNoRegressionDetectionPipeline(unittest.TestCase):
    def test_alert_manager_still_called(self):
        system = _make_system()
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.alert_manager.on_detection.assert_called_once_with(False)

    def test_sender_not_called_on_raw_no_helmet_before_confirmation(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.88}
        system.process_frame(_frame())
        system.sender.send_alert.assert_not_called()

    def test_sender_not_called_on_helmet(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        system.sender.send_alert.assert_not_called()

    def test_dashboard_detection_still_updated(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        snap = system.dashboard.snapshot()["detection"]
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], "helmet")

    def test_prev_bboxes_updated_after_frame(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([10, 20, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        self.assertEqual(len(system._prev_bboxes), 1)
        self.assertEqual(system._prev_bboxes[0], [10, 20, 100, 200])

    def test_prev_bboxes_cleared_when_no_persons(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([10, 20, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())

        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        self.assertEqual(system._prev_bboxes, [])


# ---------------------------------------------------------------------------
# Same-frame duplicate protection
# ---------------------------------------------------------------------------

class TestSameFrameDuplicateProtection(unittest.TestCase):
    def test_identical_bbox_twice_in_one_frame_counted_once(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([0, 0, 100, 200]),
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": "helmet", "confidence": 0.9},
            {"label": "helmet", "confidence": 0.9},
        ]

        system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 1)

    def test_overlapping_bboxes_in_one_frame_counted_once(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([5, 5, 95, 195]),
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": "helmet", "confidence": 0.9},
            {"label": "no_helmet", "confidence": 0.85},
        ]

        system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 1)
        self.assertEqual(_snap_stats(system)["helmet"], 1)

    def test_non_overlapping_bboxes_in_one_frame_counted_separately(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([400, 0, 100, 200]),
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": "helmet", "confidence": 0.9},
            {"label": "no_helmet", "confidence": 0.85},
        ]

        system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 2)

    def test_three_duplicates_in_one_frame_counted_once(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([0, 0, 100, 200]),
            _person([2, 2, 98, 198]),
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": "helmet", "confidence": 0.9},
            {"label": "helmet", "confidence": 0.9},
            {"label": "helmet", "confidence": 0.9},
        ]

        system.process_frame(_frame())

        self.assertEqual(_snap_stats(system)["inspected"], 1)


# ---------------------------------------------------------------------------
# KST initialization and rollover
# ---------------------------------------------------------------------------

class TestKstInitialization(unittest.TestCase):
    def test_initial_stats_date_matches_kst_today(self):
        ds = DashboardState()
        kst_today = datetime.now(_KST).date().isoformat()
        self.assertEqual(ds.snapshot()["statistics"]["date"], kst_today)

    def test_rollover_date_stored_is_kst(self):
        ds = DashboardState()
        yesterday = datetime.now(_KST).date() - timedelta(days=1)
        with ds._lock:
            ds._stats_date = yesterday
        ds.update_statistics(inspected_delta=1)
        kst_today = datetime.now(_KST).date().isoformat()
        self.assertEqual(ds.snapshot()["statistics"]["date"], kst_today)

    def test_statistics_timezone_constant_is_kst(self):
        import zoneinfo
        self.assertEqual(STATISTICS_TIMEZONE, zoneinfo.ZoneInfo("Asia/Seoul"))

    def test_utc_kst_boundary_date_correctness(self):
        utc_dt = datetime(2026, 8, 11, 15, 30, 0, tzinfo=timezone.utc)
        kst_date = utc_dt.astimezone(_KST).date()
        self.assertEqual(kst_date.isoformat(), "2026-08-12")

    def test_update_statistics_rollover_uses_kst(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=5)
        yesterday = datetime.now(_KST).date() - timedelta(days=1)
        with ds._lock:
            ds._stats_date = yesterday
        ds.update_statistics(inspected_delta=1)
        snap = ds.snapshot()["statistics"]
        self.assertEqual(snap["inspected"], 1)
        self.assertEqual(snap["date"], datetime.now(_KST).date().isoformat())

    def test_snapshot_after_kst_midnight_resets_stale_counters(self):
        ds = DashboardState()
        ds.update_statistics(inspected_delta=10, warnings_delta=3)
        yesterday = datetime.now(_KST).date() - timedelta(days=1)
        with ds._lock:
            ds._stats_date = yesterday
        snap = ds.snapshot()["statistics"]
        self.assertEqual(snap["inspected"], 0)
        self.assertEqual(snap["warnings"], 0)
        self.assertEqual(snap["date"], datetime.now(_KST).date().isoformat())

    def test_snapshot_date_after_rollover_is_kst(self):
        ds = DashboardState()
        yesterday = datetime.now(_KST).date() - timedelta(days=1)
        with ds._lock:
            ds._stats_date = yesterday
        snap = ds.snapshot()["statistics"]
        self.assertEqual(snap["date"], datetime.now(_KST).date().isoformat())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Warning integration tests
# ---------------------------------------------------------------------------

class TestWarningIntegration(unittest.TestCase):
    def test_on_no_helmet_alert_increments_warnings_once(self):
        system = _make_system()
        system.on_no_helmet_alert()
        self.assertEqual(_snap_stats(system)["warnings"], 1)

    def test_ordinary_no_helmet_frame_before_trigger_does_not_increment_warnings(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.88}
        for _ in range(2):
            system.process_frame(_frame())
        self.assertEqual(_snap_stats(system)["warnings"], 0)

    def test_on_no_helmet_alert_multiple_triggers_count_each(self):
        system = _make_system()
        system.on_no_helmet_alert()
        system.on_no_helmet_alert()
        system.on_no_helmet_alert()
        self.assertEqual(_snap_stats(system)["warnings"], 3)

    def test_warning_increment_does_not_affect_helmet_detection_behavior(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.88}
        system.process_frame(_frame())
        system.alert_manager.on_detection.assert_called()

    def test_inspected_equals_helmet_plus_no_helmet_invariant(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([300, 0, 100, 200]),
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": "helmet", "confidence": 0.9},
            {"label": "no_helmet", "confidence": 0.85},
        ]
        system.process_frame(_frame())
        snap = _snap_stats(system)
        self.assertEqual(snap["inspected"], snap["helmet"] + snap["no_helmet"])

    def test_accepted_worker_counts_once_across_repeated_frames(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        for _ in range(60):
            system.process_frame(_frame())
        self.assertEqual(_snap_stats(system)["inspected"], 1)

    def test_worker_reentry_behavior_remains_unchanged(self):
        system = _make_system()
        bbox = [0, 0, 100, 200]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.person_detector.detect.return_value = [_person(bbox)]
        system.process_frame(_frame())
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [_person(bbox)]
        system.process_frame(_frame())
        self.assertEqual(_snap_stats(system)["inspected"], 2)


class TestModuleEntrypoint(unittest.TestCase):
    def test_module_importable_as_test_daily_statistics(self):
        import importlib
        mod = importlib.import_module("mpu.tests.test_daily_statistics")
        self.assertIsNotNone(mod)

    def test_test_classes_discoverable(self):
        import unittest
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName("mpu.tests.test_daily_statistics")
        self.assertGreater(suite.countTestCases(), 0)


if __name__ == "__main__":
    unittest.main()
