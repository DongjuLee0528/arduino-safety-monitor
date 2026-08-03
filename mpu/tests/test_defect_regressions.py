"""
Regression tests for two confirmed production defects.

Defect 1 – STOP Must Always Work in AUTO (arduino/comm.h)
Defect 2 – Dashboard worker_present Must Use Validated Detections (mpu/main.py)

Arduino tests are static source-text structural tests; they do not require
native MCU execution.  Python tests exercise the live HelmetDetectionSystem
process_frame() path with all I/O mocked.
"""

import os
import unittest
from unittest.mock import MagicMock

import numpy as np

from mpu.dashboard_state import DashboardState, HelmetResult, EventType
from mpu.main import HelmetDetectionSystem, _EventSuppressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comm_h_text():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "arduino", "comm.h")
    )
    with open(path) as f:
        return f.read()


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


def _valid_person(x=0, y=0, w=100, h=200):
    return {"bbox": [x, y, w, h], "confidence": 0.9}


def _snap_detection(system):
    return system.dashboard.snapshot()["detection"]


def _snap_stats(system):
    return system.dashboard.snapshot()["statistics"]


def _event_messages(system):
    return [e["message"] for e in system.dashboard.snapshot()["events"]]


# ---------------------------------------------------------------------------
# Defect 1 – STOP priority in AUTO mode (Arduino comm.h structural tests)
# ---------------------------------------------------------------------------

class TestStopInAutoStructural(unittest.TestCase):
    """
    Verify the STOP-in-AUTO fix at the Arduino source level.

    All tests are static structural checks on the comm.h source text.
    They confirm the required ordering and logic without MCU execution.
    """

    def setUp(self):
        self.src = _comm_h_text()

    def _motor_block(self):
        start = self.src.find('} else if (strcmp(cmd, "motor") == 0) {')
        end = self.src.find("\n        } else if (strcmp(cmd", start + 1)
        if end == -1:
            end = self.src.find("\n        } else {", start + 1)
        return self.src[start:end]

    def test_01_stop_accepted_in_auto_mode(self):
        block = self._motor_block()
        self.assertIn("mc != MOVE_STOP", block,
                      "AUTO rejection must exempt MOVE_STOP")

    def test_02_auto_rejection_is_after_direction_parse(self):
        block = self._motor_block()
        direction_parse_pos = block.find("jsonGetString")
        auto_reject_pos = block.find("CMD_NOT_ALLOWED_IN_AUTO")
        self.assertGreater(auto_reject_pos, direction_parse_pos,
                           "Direction must be parsed before AUTO rejection check")

    def test_03_auto_rejection_is_after_move_cmd_assignment(self):
        block = self._motor_block()
        move_assign_pos = block.find("MOVE_FORWARD")
        auto_reject_pos = block.find("CMD_NOT_ALLOWED_IN_AUTO")
        self.assertGreater(auto_reject_pos, move_assign_pos,
                           "mc must be assigned before AUTO rejection check")

    def test_04_auto_forward_remains_rejected(self):
        block = self._motor_block()
        self.assertIn("CMD_NOT_ALLOWED_IN_AUTO", block)
        self.assertIn("mc != MOVE_STOP", block,
                      "Only MOVE_STOP is exempted; forward/backward/left/right remain rejected")

    def test_05_auto_backward_remains_rejected(self):
        block = self._motor_block()
        self.assertIn("mc != MOVE_STOP", block)
        self.assertNotIn("mc != MOVE_BACKWARD", block,
                         "BACKWARD is not individually exempted; MOVE_STOP guard covers it")

    def test_06_auto_left_remains_rejected(self):
        block = self._motor_block()
        self.assertNotIn("mc != MOVE_LEFT", block)

    def test_07_auto_right_remains_rejected(self):
        block = self._motor_block()
        self.assertNotIn("mc != MOVE_RIGHT", block)

    def test_08_stop_forces_manual_mode(self):
        block = self._motor_block()
        stop_branch_pos = block.find("if (mc == MOVE_STOP)")
        self.assertNotEqual(stop_branch_pos, -1,
                            "Explicit MOVE_STOP branch must exist")
        stop_snippet = block[stop_branch_pos:stop_branch_pos + 200]
        self.assertIn("MODE_MANUAL", stop_snippet,
                      "STOP must force MODE_MANUAL")

    def test_09_stop_clears_pending_move(self):
        block = self._motor_block()
        stop_branch_pos = block.find("if (mc == MOVE_STOP)")
        stop_snippet = block[stop_branch_pos:stop_branch_pos + 200]
        self.assertIn("MOVE_STOP", stop_snippet)
        self.assertIn("_pendingMove", stop_snippet)

    def test_10_stop_resets_pending_speed_to_default(self):
        block = self._motor_block()
        stop_branch_pos = block.find("if (mc == MOVE_STOP)")
        stop_snippet = block[stop_branch_pos:stop_branch_pos + 200]
        self.assertIn("MOTOR_SPEED_DEFAULT", stop_snippet,
                      "STOP must reset pending speed to default")

    def test_11_stop_ack_is_returned(self):
        block = self._motor_block()
        self.assertIn("motor_ack", block,
                      "motor_ack must be sent for stop (shared ack path)")
        ack_pos = block.rfind("motor_ack")
        reject_pos = block.find("CMD_NOT_ALLOWED_IN_AUTO")
        self.assertGreater(ack_pos, reject_pos,
                           "ACK emission must follow the AUTO rejection block")

    def test_12_comm_manager_has_no_direct_motor_controller(self):
        src = self.src
        self.assertNotIn("MotorController", src,
                         "CommunicationManager must not reference MotorController")
        self.assertNotIn("_motor", src,
                         "CommunicationManager must not hold a _motor member")


# ---------------------------------------------------------------------------
# Defect 2 – worker_present uses validated detections (Python runtime tests)
# ---------------------------------------------------------------------------

class TestWorkerPresentValidated(unittest.TestCase):

    def test_13_empty_detector_result_worker_not_present(self):
        system = _make_system()
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        self.assertFalse(_snap_detection(system)["worker_present"])

    def test_14_one_malformed_detection_worker_not_present(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [float("nan"), 0, 100, 200], "confidence": 0.9}
        ]
        system.process_frame(_frame())
        self.assertFalse(_snap_detection(system)["worker_present"])

    def test_15_multiple_malformed_detections_worker_not_present(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [float("nan"), 0, 100, 200], "confidence": 0.9},
            {"bbox": [True, 0, 100, 200], "confidence": 0.8},
            "not_a_dict",
        ]
        system.process_frame(_frame())
        self.assertFalse(_snap_detection(system)["worker_present"])

    def test_16_malformed_then_valid_worker_present(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [float("nan"), 0, 100, 200], "confidence": 0.9},
            _valid_person(0, 0, 100, 200),
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.9
        }
        system.process_frame(_frame())
        self.assertTrue(_snap_detection(system)["worker_present"])

    def test_17_valid_then_malformed_worker_still_present(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _valid_person(0, 0, 100, 200),
            {"bbox": [float("nan"), 0, 100, 200], "confidence": 0.9},
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.9
        }
        system.process_frame(_frame())
        self.assertTrue(_snap_detection(system)["worker_present"])

    def test_18_valid_bbox_malformed_classifier_still_worker_present(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_valid_person()]
        system.helmet_classifier.predict.return_value = {
            "label": "garbage_label", "confidence": 0.9
        }
        system.process_frame(_frame())
        self.assertTrue(_snap_detection(system)["worker_present"])

    def test_19_malformed_only_no_stat_increment(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [float("inf"), 0, 100, 200], "confidence": 0.9},
        ]
        system.process_frame(_frame())
        stats = _snap_stats(system)
        self.assertEqual(stats["inspected"], 0)
        self.assertEqual(stats["helmet"], 0)
        self.assertEqual(stats["no_helmet"], 0)

    def test_20_malformed_only_no_detection_events(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [float("nan"), 0, 100, 200], "confidence": 0.9},
        ]
        system.process_frame(_frame())
        messages = _event_messages(system)
        self.assertFalse(
            any("detected" in m.lower() for m in messages),
            f"No detection events expected; got: {messages}"
        )

    def test_21_malformed_only_no_alert_triggered(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [True, 0, 100, 200], "confidence": 0.9},
        ]
        system.process_frame(_frame())
        system.sender.send_alert.assert_not_called()
        system.alert_manager.on_detection.assert_called_once_with(False)

    def test_22_valid_helmet_path_unchanged(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_valid_person()]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.95
        }
        system.process_frame(_frame())
        snap = _snap_detection(system)
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], "helmet")
        system.sender.send_alert.assert_not_called()
        system.alert_manager.on_detection.assert_called_once_with(False)

    def test_22b_valid_no_helmet_path_unchanged(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_valid_person()]
        system.helmet_classifier.predict.return_value = {
            "label": "no_helmet", "confidence": 0.88
        }
        system.process_frame(_frame())
        snap = _snap_detection(system)
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], "no_helmet")
        system.sender.send_alert.assert_called_once()
        system.alert_manager.on_detection.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
