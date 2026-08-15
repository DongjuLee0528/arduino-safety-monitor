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
        start = self.src.find("String rpcMotor(")
        self.assertNotEqual(start, -1)
        return self.src[start:start + 1400]

    def test_01_stop_accepted_in_auto_mode(self):
        block = self._motor_block()
        self.assertIn("mc != MOVE_STOP", block,
                      "AUTO rejection must exempt MOVE_STOP")

    def test_02_auto_rejection_is_after_direction_parse(self):
        block = self._motor_block()
        direction_parse_pos = block.find("_movementFromDirection(direction)")
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
        self.assertIn("_motorAck", block,
                      "motor_ack must be sent for stop (shared ack path)")
        ack_pos = block.rfind("_motorAck")
        reject_pos = block.find("CMD_NOT_ALLOWED_IN_AUTO")
        self.assertGreater(ack_pos, reject_pos,
                           "ACK emission must follow the AUTO rejection block")

    def test_12_comm_manager_has_no_direct_motor_controller(self):
        src = self.src
        self.assertNotIn("MotorController", src,
                         "CommunicationManager must not reference MotorController")
        private_members = src[src.find("private:"):src.find("public:")]
        self.assertNotIn("MotorController*", private_members,
                         "CommunicationManager must not hold a MotorController pointer")
        self.assertNotIn("_motor;", private_members,
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


# ---------------------------------------------------------------------------
# Defect 3 – STOP Durability (comm.h structural tests)
# ---------------------------------------------------------------------------

def _comm_h_src():
    return _comm_h_text()


class TestStopDurabilityStructural(unittest.TestCase):
    """
    Verify the STOP-durability latch at the Arduino source level.

    All tests are static structural checks on the comm.h source text.
    """

    def setUp(self):
        self.src = _comm_h_src()

    def test_23_stop_latched_flag_declared(self):
        self.assertIn("_stopLatched", self.src,
                      "_stopLatched member must exist in CommunicationManager")

    def test_24_stop_sets_stop_latched(self):
        motor_start = self.src.find("String rpcMotor(")
        block = self.src[motor_start:motor_start + 1400]
        stop_branch_pos = block.find("if (mc == MOVE_STOP)")
        self.assertNotEqual(stop_branch_pos, -1)
        stop_snippet = block[stop_branch_pos:stop_branch_pos + 300]
        self.assertIn("_stopLatched  = true", stop_snippet,
                      "STOP branch must set _stopLatched = true")

    def test_25_non_stop_move_guarded_by_stop_latched(self):
        motor_start = self.src.find("String rpcMotor(")
        block = self.src[motor_start:motor_start + 1400]
        self.assertIn("!_stopLatched", block,
                      "Non-STOP move assignment must be guarded by !_stopLatched")

    def test_26_mode_auto_mode_assignment_gated_by_stop_latched(self):
        mode_start = self.src.find("String rpcMode(")
        block = self.src[mode_start:mode_start + 900]
        auto_pos   = block.find('value == "auto"')
        self.assertNotEqual(auto_pos, -1)
        auto_branch = block[auto_pos:auto_pos + 300]
        guard_pos   = auto_branch.find("!_stopLatched")
        mode_assign = auto_branch.find("_mode         = MODE_AUTO")
        self.assertNotEqual(guard_pos,   -1, "mode=auto branch must contain !_stopLatched guard")
        self.assertNotEqual(mode_assign, -1, "mode=auto branch must assign _mode = MODE_AUTO")
        self.assertLess(guard_pos, mode_assign,
                        "!_stopLatched guard must appear before _mode = MODE_AUTO assignment")

    def test_27_consume_clears_stop_latched(self):
        consume_pos = self.src.find("MovementCmd consumePendingMove()")
        self.assertNotEqual(consume_pos, -1)
        consume_snippet = self.src[consume_pos:consume_pos + 300]
        self.assertIn("_stopLatched  = false", consume_snippet,
                      "consumePendingMove() must clear _stopLatched")

    def test_28_stop_latched_initialised_false(self):
        ctor_pos = self.src.find("CommunicationManager()")
        self.assertNotEqual(ctor_pos, -1)
        ctor_snippet = self.src[ctor_pos:ctor_pos + 400]
        self.assertIn("_stopLatched(false)", ctor_snippet,
                      "_stopLatched must be initialised to false in the constructor")

    def test_29_auto_forward_still_rejected(self):
        self.assertIn("CMD_NOT_ALLOWED_IN_AUTO", self.src,
                      "AUTO rejection for non-STOP moves must still exist")
        self.assertIn("mc != MOVE_STOP", self.src,
                      "AUTO rejection must still exempt only MOVE_STOP")

    def test_30_safe_reset_sets_stop_latched(self):
        sr_pos = self.src.find("String rpcSafeReset()")
        self.assertNotEqual(sr_pos, -1)
        sr_snippet = self.src[sr_pos:sr_pos + 300]
        self.assertIn("_stopLatched  = true", sr_snippet,
                      "safe_reset must also set _stopLatched")


# ---------------------------------------------------------------------------
# Defect 4 – Ultrasonic Freshness (ultrasonic.h structural tests)
# ---------------------------------------------------------------------------

def _ultrasonic_h_text():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "arduino", "ultrasonic.h")
    )
    with open(path) as f:
        return f.read()


def _config_h_text():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "arduino", "config.h")
    )
    with open(path) as f:
        return f.read()


class TestUltrasonicFreshnessStructural(unittest.TestCase):
    """
    Verify the ultrasonic freshness fix at the Arduino source level.
    """

    def setUp(self):
        self.src = _ultrasonic_h_text()
        self.cfg = _config_h_text()

    def test_31_stale_timeout_defined_in_config(self):
        self.assertIn("SENSOR_STALE_TIMEOUT_MS", self.cfg,
                      "SENSOR_STALE_TIMEOUT_MS must be defined in config.h")

    def test_32_last_valid_time_array_declared(self):
        self.assertIn("_lastValidTime", self.src,
                      "_lastValidTime array must be declared in UltrasonicSensor")

    def test_33_last_valid_time_set_on_valid_reading(self):
        measure_pos = self.src.find("void measureSensor(int sensor)")
        self.assertNotEqual(measure_pos, -1)
        measure_end = self.src.find("\n    }", measure_pos)
        block = self.src[measure_pos:measure_end + 10]
        self.assertIn("_lastValidTime", block,
                      "_lastValidTime must be updated inside measureSensor() on a valid reading")

    def test_34_distance_available_checks_freshness(self):
        avail_pos = self.src.find("bool distanceAvailable(int sensor)")
        self.assertNotEqual(avail_pos, -1)
        avail_end = self.src.find("\n    }", avail_pos)
        block = self.src[avail_pos:avail_end + 10]
        self.assertIn("SENSOR_STALE_TIMEOUT_MS", block,
                      "distanceAvailable() must check SENSOR_STALE_TIMEOUT_MS")
        self.assertIn("_lastValidTime", block,
                      "distanceAvailable() must reference _lastValidTime")

    def test_35_distance_available_still_guards_zero_count(self):
        avail_pos = self.src.find("bool distanceAvailable(int sensor)")
        avail_end = self.src.find("\n    }", avail_pos)
        block = self.src[avail_pos:avail_end + 10]
        self.assertIn("measureCount[sensor] == 0", block,
                      "distanceAvailable() must still return false when measureCount is 0")

    def test_36_last_valid_time_initialised_to_zero(self):
        ctor_pos = self.src.find("UltrasonicSensor(int ft")
        self.assertNotEqual(ctor_pos, -1)
        ctor_end = self.src.find("\n    }", ctor_pos)
        ctor_snippet = self.src[ctor_pos:ctor_end + 10]
        self.assertIn("_lastValidTime", ctor_snippet,
                      "_lastValidTime must be initialised to 0 in the constructor")

    def test_37_measurement_algorithm_unchanged(self):
        self.assertIn("pulseIn(echoPin, HIGH, ULTRASONIC_TIMEOUT_US)", self.src,
                      "pulseIn call must be unchanged")
        self.assertIn("duration * 0.034f / 2.0f", self.src,
                      "Distance formula must be unchanged")

    def test_38_filtering_unchanged(self):
        self.assertIn("measureCount[sensor] % ULTRASONIC_SAMPLES", self.src,
                      "Rolling-average ring-buffer indexing must be unchanged")
        self.assertIn("ULTRASONIC_SAMPLES", self.src)

    def test_39_valid_flag_declared(self):
        self.assertIn("_valid", self.src,
                      "_valid[] explicit validity array must be declared")

    def test_40_valid_set_true_on_valid_reading(self):
        measure_pos = self.src.find("void measureSensor(int sensor)")
        self.assertNotEqual(measure_pos, -1)
        measure_end = self.src.find("\n    }", measure_pos)
        block = self.src[measure_pos:measure_end + 10]
        valid_true_pos = block.find("_valid[sensor]         = true")
        self.assertNotEqual(valid_true_pos, -1,
                            "_valid[sensor] must be set true on a valid reading")

    def test_41_valid_set_false_on_timeout(self):
        measure_pos = self.src.find("void measureSensor(int sensor)")
        measure_end = self.src.find("\n    }", measure_pos)
        block = self.src[measure_pos:measure_end + 10]
        else_pos       = block.find("} else {")
        self.assertNotEqual(else_pos, -1,
                            "measureSensor must have an else branch for invalid/timeout pulse")
        else_branch    = block[else_pos:else_pos + 80]
        self.assertIn("_valid[sensor] = false", else_branch,
                      "_valid[sensor] must be set false in the timeout/invalid else branch")

    def test_42_distance_available_requires_valid_flag(self):
        avail_pos = self.src.find("bool distanceAvailable(int sensor)")
        self.assertNotEqual(avail_pos, -1)
        avail_end = self.src.find("\n    }", avail_pos)
        block = self.src[avail_pos:avail_end + 10]
        self.assertIn("_valid[sensor]", block,
                      "distanceAvailable() must check _valid[sensor]")

    def test_43_valid_initialised_false_in_constructor(self):
        ctor_pos = self.src.find("UltrasonicSensor(int ft")
        self.assertNotEqual(ctor_pos, -1)
        ctor_end = self.src.find("\n    }", ctor_pos)
        ctor_snippet = self.src[ctor_pos:ctor_end + 10]
        self.assertIn("_valid[i]         = false", ctor_snippet,
                      "_valid must be initialised to false in the constructor")


if __name__ == "__main__":
    unittest.main()
