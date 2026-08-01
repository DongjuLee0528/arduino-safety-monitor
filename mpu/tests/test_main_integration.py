"""
Integration tests for HelmetDetectionSystem <-> DashboardState wiring.

Tests verify that:
  - connection updates reach DashboardState
  - detection updates reach DashboardState
  - no regression of existing alert / hardware logic
  - DashboardState snapshot reflects the most recent runtime state

All external I/O (camera, bridge_rpc, sender, person_detector,
helmet_classifier) is mocked so the test suite runs without hardware.
"""

import unittest
from datetime import timezone
from unittest.mock import MagicMock, patch, PropertyMock

from mpu.dashboard_state import DashboardState, HelmetResult, RobotConnection
from mpu.main import HelmetDetectionSystem, _EventSuppressor


def _make_system():
    """
    Construct a HelmetDetectionSystem with all external dependencies mocked.
    Returns the system instance.
    """
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


class TestConnectionIntegration(unittest.TestCase):
    def test_connect_sets_dashboard_online(self):
        system = _make_system()
        system.bridge_rpc.connect.return_value = None
        # Raise on the first camera read so the loop exits immediately
        # after the connection path (bridge_rpc.connect + dashboard update).
        system.camera.capture_frame.side_effect = RuntimeError("stop loop")
        system.person_detector.detect.return_value = []

        # Spy on update_connection while letting the real method execute,
        # so we can verify call ordering independently of stop()'s offline update.
        recorded_online_args = []
        real_update = system.dashboard.update_connection
        def _spy(online, **kw):
            recorded_online_args.append(online)
            return real_update(online=online, **kw)
        system.dashboard.update_connection = _spy

        system.start()

        system.bridge_rpc.connect.assert_called_once()
        # The first update_connection call must have been online=True (from start()).
        # stop() appends a second call with online=False.
        self.assertTrue(len(recorded_online_args) >= 1)
        self.assertTrue(recorded_online_args[0])

    def test_stop_sets_dashboard_offline(self):
        system = _make_system()
        system.bridge_rpc.connect.return_value = None
        system.dashboard.update_connection(online=True)

        system.stop()

        snap = system.dashboard.snapshot()
        self.assertEqual(snap["connection"]["status"], "offline")

    def test_connection_timestamp_is_utc_aware(self):
        system = _make_system()
        system.dashboard.update_connection(online=True)
        ts_str = system.dashboard.snapshot()["connection"]["updated_at"]
        self.assertIsNotNone(ts_str)
        self.assertIn("+00:00", ts_str)

    def test_stop_called_twice_stays_offline(self):
        system = _make_system()
        system.dashboard.update_connection(online=True)
        system.stop()
        system.stop()
        self.assertEqual(system.dashboard.snapshot()["connection"]["status"], "offline")


class TestDetectionIntegration(unittest.TestCase):
    def _frame(self):
        import numpy as np
        return np.zeros((480, 640, 3), dtype="uint8")

    def test_no_persons_sets_worker_not_present(self):
        system = _make_system()
        system.person_detector.detect.return_value = []

        system.process_frame(self._frame())

        snap = system.dashboard.snapshot()["detection"]
        self.assertFalse(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], HelmetResult.UNKNOWN.value)
        self.assertIsNone(snap["bbox"])

    def test_person_with_helmet_sets_helmet_result(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [10, 20, 80, 160], "confidence": 0.9}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.95
        }

        system.process_frame(self._frame())

        snap = system.dashboard.snapshot()["detection"]
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], "helmet")
        self.assertEqual(snap["bbox"], [10.0, 20.0, 80.0, 160.0])

    def test_person_without_helmet_sets_no_helmet_result(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [5, 10, 100, 200], "confidence": 0.85}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "no_helmet", "confidence": 0.88
        }
        system.sender.send_alert.side_effect = Exception("network unavailable")

        system.process_frame(self._frame())

        snap = system.dashboard.snapshot()["detection"]
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], "no_helmet")

    def test_mixed_persons_no_helmet_wins(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [0, 0, 50, 100], "confidence": 0.9},
            {"bbox": [200, 0, 50, 100], "confidence": 0.8},
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": "helmet", "confidence": 0.9},
            {"label": "no_helmet", "confidence": 0.7},
        ]

        system.process_frame(self._frame())

        snap = system.dashboard.snapshot()["detection"]
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], "no_helmet")

    def test_bbox_stored_as_list_of_floats(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [7, 13, 55, 110], "confidence": 0.9}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.9
        }

        system.process_frame(self._frame())

        bbox = system.dashboard.snapshot()["detection"]["bbox"]
        self.assertIsInstance(bbox, list)
        self.assertEqual(len(bbox), 4)
        for v in bbox:
            self.assertIsInstance(v, float)

    def test_detection_updated_at_is_utc_aware(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [0, 0, 40, 80], "confidence": 0.9}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.9
        }

        system.process_frame(self._frame())

        ts_str = system.dashboard.snapshot()["detection"]["updated_at"]
        self.assertIsNotNone(ts_str)
        self.assertIn("+00:00", ts_str)

    def test_successive_frames_update_detection(self):
        system = _make_system()

        system.person_detector.detect.return_value = [
            {"bbox": [0, 0, 40, 80], "confidence": 0.9}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.9
        }
        system.process_frame(self._frame())
        self.assertTrue(system.dashboard.snapshot()["detection"]["worker_present"])

        system.person_detector.detect.return_value = []
        system.process_frame(self._frame())
        self.assertFalse(system.dashboard.snapshot()["detection"]["worker_present"])

    def test_snapshot_is_json_serializable_after_detection(self):
        import json
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [10, 20, 80, 160], "confidence": 0.9}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "no_helmet", "confidence": 0.88
        }
        system.sender.send_alert.side_effect = Exception("offline")

        system.process_frame(self._frame())

        snap = system.dashboard.snapshot()
        serialized = json.dumps(snap)
        self.assertIsInstance(serialized, str)


class TestNoRegressionExistingBehavior(unittest.TestCase):
    def _frame(self):
        import numpy as np
        return np.zeros((480, 640, 3), dtype="uint8")

    def test_alert_manager_still_called(self):
        system = _make_system()
        system.person_detector.detect.return_value = []

        system.process_frame(self._frame())

        system.alert_manager.on_detection.assert_called_once_with(False)

    def test_sender_called_on_no_helmet(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [0, 0, 40, 80], "confidence": 0.9}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "no_helmet", "confidence": 0.88
        }

        import numpy as np
        frame = np.zeros((480, 640, 3), dtype="uint8")
        system.process_frame(frame)

        system.sender.send_alert.assert_called_once()

    def test_sender_not_called_on_helmet(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [0, 0, 40, 80], "confidence": 0.9}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.95
        }

        system.process_frame(self._frame())

        system.sender.send_alert.assert_not_called()

    def test_bridge_disconnect_called_in_stop(self):
        system = _make_system()
        system.stop()
        system.bridge_rpc.disconnect.assert_called_once()

    def test_dashboard_attribute_exists_after_init(self):
        with patch("mpu.main.CameraCapture"), \
             patch("mpu.main.PersonDetector"), \
             patch("mpu.main.HelmetClassifier"), \
             patch("mpu.main.BridgeRPC"), \
             patch("mpu.main.Sender"), \
             patch("mpu.main.AlertManager"):
            system = HelmetDetectionSystem(port="/dev/null", server_url="http://localhost")
        self.assertIsInstance(system.dashboard, DashboardState)

    def test_empty_crop_skipped_no_crash(self):
        import numpy as np
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [0, 0, 0, 0], "confidence": 0.9}
        ]
        frame = np.zeros((480, 640, 3), dtype="uint8")

        system.process_frame(frame)

        snap = system.dashboard.snapshot()["detection"]
        # detect() returned one person so worker_present is True; the crop is
        # empty so no classifier call is made and helmet_result stays UNKNOWN.
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], HelmetResult.UNKNOWN.value)


class TestModeAndMovementDefaults(unittest.TestCase):
    def test_mode_defaults_unknown(self):
        system = _make_system()
        self.assertEqual(system.dashboard.snapshot()["mode"], "unknown")

    def test_movement_defaults_stopped(self):
        system = _make_system()
        self.assertEqual(system.dashboard.snapshot()["movement"], "stopped")

    def test_distances_default_none(self):
        system = _make_system()
        d = system.dashboard.snapshot()["distances"]
        self.assertIsNone(d["front"])
        self.assertIsNone(d["rear"])
        self.assertIsNone(d["left"])
        self.assertIsNone(d["right"])


if __name__ == "__main__":
    unittest.main()
