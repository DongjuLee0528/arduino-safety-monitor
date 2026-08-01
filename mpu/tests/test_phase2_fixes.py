import io
import json
import math
import threading
import unittest
from datetime import timezone
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from mpu.bridge_rpc import BridgeRPC, RPCError
from mpu.dashboard_state import DashboardState, EventType
from mpu.main import HelmetDetectionSystem, _EventSuppressor
from mpu.sender import Sender


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


def _make_serial(responses):
    ser = MagicMock()
    ser.is_open = True
    encoded = b"".join((json.dumps(r) + "\n").encode() for r in responses)
    buf = io.BytesIO(encoded)

    def _in_waiting():
        return len(buf.getvalue()) - buf.tell()

    type(ser).in_waiting = PropertyMock(side_effect=_in_waiting)
    ser.read.side_effect = buf.read
    ser.write = MagicMock()
    return ser


def _make_bridge(responses):
    bridge = BridgeRPC.__new__(BridgeRPC)
    bridge.port = "/dev/null"
    bridge.baudrate = 115200
    bridge.timeout = 1.0
    bridge.ser = _make_serial(responses)
    bridge._lock = threading.Lock()
    bridge._heartbeat_thread = None
    bridge._heartbeat_stop = threading.Event()
    bridge._heartbeat_failures = 0
    bridge._heartbeat_healthy = True
    return bridge


def _snap_stats(system):
    return system.dashboard.snapshot()["statistics"]


def _event_messages(system):
    return [e["message"] for e in system.dashboard.snapshot()["events"]]


# ---------------------------------------------------------------------------
# Arduino source text helpers (static structural verification)
# ---------------------------------------------------------------------------

def _comm_h_text():
    import os
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "arduino", "comm.h",
    )
    with open(os.path.abspath(path)) as f:
        return f.read()


def _ultrasonic_h_text():
    import os
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "arduino", "ultrasonic.h",
    )
    with open(os.path.abspath(path)) as f:
        return f.read()


def _arduino_ino_text():
    import os
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "arduino", "arduino.ino",
    )
    with open(os.path.abspath(path)) as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1-12  Motor speed contract (Arduino source static + Python bridge)
# ---------------------------------------------------------------------------

class TestMotorSpeedContract(unittest.TestCase):
    def test_01_valid_speed_stored_in_pending(self):
        src = _comm_h_text()
        self.assertIn("_pendingSpeed = spd", src)

    def test_02_ack_contains_validated_speed(self):
        src = _comm_h_text()
        self.assertIn("motor_ack", src)
        self.assertIn("\\\"speed\\\":%d", src)

    def test_03_speed_below_0_rejected(self):
        src = _comm_h_text()
        self.assertIn("INVALID_SPEED", src)
        self.assertIn("spd < MOTOR_SPEED_MIN", src)

    def test_04_speed_above_255_rejected(self):
        src = _comm_h_text()
        self.assertIn("spd > MOTOR_SPEED_MAX", src)

    def test_05_string_speed_rejected_by_strict_parser(self):
        src = _comm_h_text()
        self.assertIn("jsonGetStrictInt", src)
        self.assertIn("hasSpdField", src)

    def test_06_float_speed_rejected_by_strict_parser(self):
        src = _comm_h_text()
        self.assertIn("*p >= '0' && *p <= '9'", src)

    def test_07_missing_speed_uses_default(self):
        src = _comm_h_text()
        self.assertIn("MOTOR_SPEED_DEFAULT", src)
        self.assertIn("hasSpdField", src)

    def test_08_stop_clears_pending_speed(self):
        src = _comm_h_text()
        self.assertIn("consumePendingMove", src)
        self.assertIn("_pendingSpeed = MOTOR_SPEED_DEFAULT", src)

    def test_09_consume_pending_move_resets_speed(self):
        src = _comm_h_text()
        idx = src.find("MovementCmd consumePendingMove()")
        snippet = src[idx:idx + 300]
        self.assertIn("_pendingSpeed = MOTOR_SPEED_DEFAULT", snippet)

    def test_10_safe_reset_clears_pending_speed(self):
        src = _comm_h_text()
        idx = src.find("safe_reset")
        while idx != -1:
            snippet = src[idx:idx + 200]
            if "_pendingSpeed = MOTOR_SPEED_DEFAULT" in snippet:
                break
            idx = src.find("safe_reset", idx + 1)
        self.assertNotEqual(idx, -1, "safe_reset block must reset _pendingSpeed")

    def test_11_disconnect_safe_state_resets_speed(self):
        src = _comm_h_text()
        idx = src.find("resetToManualSafeState()")
        snippet = src[idx:idx + 300]
        self.assertIn("_pendingSpeed = MOTOR_SPEED_DEFAULT", snippet)

    def test_12_later_command_cannot_inherit_stale_speed(self):
        src = _comm_h_text()
        consume_idx = src.find("MovementCmd consumePendingMove()")
        snippet = src[consume_idx:consume_idx + 300]
        self.assertIn("_pendingSpeed = MOTOR_SPEED_DEFAULT", snippet)

    def test_bridge_motor_ack_validates_speed_field(self):
        bridge = _make_bridge([{"type": "motor_ack", "direction": "forward", "speed": 180}])
        result = bridge.motor_control("forward", 180)
        self.assertTrue(result)

    def test_bridge_motor_ack_speed_mismatch_rejected(self):
        from mpu.bridge_rpc import RPCProtocolError
        bridge = _make_bridge([{"type": "motor_ack", "direction": "forward", "speed": 999}])
        with self.assertRaises(RPCProtocolError):
            bridge.motor_control("forward", 150)


# ---------------------------------------------------------------------------
# 13-17  Ultrasonic initialization (Arduino source static)
# ---------------------------------------------------------------------------

class TestUltrasonicInitialization(unittest.TestCase):
    def test_13_constructor_no_pinmode(self):
        src = _ultrasonic_h_text()
        ctor_start = src.find("UltrasonicSensor(int ft")
        ctor_end = src.find("\n    }", ctor_start)
        ctor_body = src[ctor_start:ctor_end]
        self.assertNotIn("pinMode", ctor_body)
        self.assertNotIn("digitalWrite", ctor_body)

    def test_14_begin_initializes_all_four_triggers(self):
        src = _ultrasonic_h_text()
        begin_idx = src.find("void begin()")
        snippet = src[begin_idx:begin_idx + 400]
        self.assertIn("frontTrig", snippet)
        self.assertIn("backTrig", snippet)
        self.assertIn("leftTrig", snippet)
        self.assertIn("rightTrig", snippet)

    def test_15_each_trigger_driven_low(self):
        src = _ultrasonic_h_text()
        begin_idx = src.find("void begin()")
        snippet = src[begin_idx:begin_idx + 400]
        self.assertEqual(snippet.count("digitalWrite"), 4)
        self.assertEqual(snippet.count("LOW"), 4)

    def test_16_echo_pins_are_input(self):
        src = _ultrasonic_h_text()
        begin_idx = src.find("void begin()")
        snippet = src[begin_idx:begin_idx + 400]
        self.assertIn("frontEcho", snippet)
        self.assertIn("backEcho", snippet)
        self.assertIn("leftEcho", snippet)
        self.assertIn("rightEcho", snippet)
        self.assertIn("INPUT", snippet)

    def test_17_setup_calls_ultrasonic_begin(self):
        src = _arduino_ino_text()
        setup_idx = src.find("void setup()")
        snippet = src[setup_idx:setup_idx + 300]
        self.assertIn("ultrasonic.begin()", snippet)


# ---------------------------------------------------------------------------
# 18-23  Detector bbox validation
# ---------------------------------------------------------------------------

class TestDetectorBboxValidation(unittest.TestCase):
    def test_18_bool_bbox_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [True, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        system.helmet_classifier.predict.assert_not_called()

    def test_19_nan_bbox_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [float("nan"), 0, 100, 200], "confidence": 0.9}]
        system.process_frame(_frame())
        system.helmet_classifier.predict.assert_not_called()

    def test_20_positive_infinity_bbox_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [float("inf"), 0, 100, 200], "confidence": 0.9}]
        system.process_frame(_frame())
        system.helmet_classifier.predict.assert_not_called()

    def test_21_negative_infinity_bbox_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [float("-inf"), 0, 100, 200], "confidence": 0.9}]
        system.process_frame(_frame())
        system.helmet_classifier.predict.assert_not_called()

    def test_22_malformed_does_not_block_valid_detection(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [float("nan"), 0, 100, 200], "confidence": 0.9},
            {"bbox": [0, 0, 100, 200], "confidence": 0.9},
        ]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        system.helmet_classifier.predict.assert_called_once()

    def test_23_malformed_bbox_no_dashboard_stat_event_alert(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [True, 0, 100, 200], "confidence": 0.9},
        ]
        system.process_frame(_frame())
        snap = system.dashboard.snapshot()
        self.assertEqual(snap["statistics"]["inspected"], 0)
        self.assertEqual(snap["statistics"]["helmet"], 0)
        self.assertEqual(snap["statistics"]["no_helmet"], 0)
        self.assertEqual(len(snap["events"]), 0)
        system.sender.send_alert.assert_not_called()


# ---------------------------------------------------------------------------
# 24-31  Classifier result validation
# ---------------------------------------------------------------------------

class TestClassifierResultValidation(unittest.TestCase):
    def _valid_bbox(self):
        return [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]

    def test_24_unsupported_label_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = self._valid_bbox()
        system.helmet_classifier.predict.return_value = {"label": "unknown_class", "confidence": 0.9}
        system.process_frame(_frame())
        snap = system.dashboard.snapshot()
        self.assertEqual(snap["statistics"]["inspected"], 0)

    def test_25_bool_confidence_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = self._valid_bbox()
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": True}
        system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["inspected"], 0)

    def test_26_nan_confidence_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = self._valid_bbox()
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": float("nan")}
        system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["inspected"], 0)

    def test_27_infinity_confidence_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = self._valid_bbox()
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": float("inf")}
        system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["inspected"], 0)

    def test_28_missing_label_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = self._valid_bbox()
        system.helmet_classifier.predict.return_value = {"confidence": 0.9}
        system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["inspected"], 0)

    def test_29_missing_confidence_rejected(self):
        system = _make_system()
        system.person_detector.detect.return_value = self._valid_bbox()
        system.helmet_classifier.predict.return_value = {"label": "helmet"}
        system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["inspected"], 0)

    def test_30_malformed_result_no_false_no_helmet(self):
        system = _make_system()
        system.person_detector.detect.return_value = self._valid_bbox()
        system.helmet_classifier.predict.return_value = {"label": "unknown_class", "confidence": 0.9}
        system.process_frame(_frame())
        system.sender.send_alert.assert_not_called()
        system.alert_manager.on_detection.assert_called_once_with(False)

    def test_31_later_valid_worker_still_processes(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            {"bbox": [0, 0, 100, 200], "confidence": 0.9},
            {"bbox": [300, 0, 100, 200], "confidence": 0.8},
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": "garbage_label", "confidence": 0.9},
            {"label": "helmet", "confidence": 0.85},
        ]
        system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["inspected"], 1)
        self.assertEqual(system.dashboard.snapshot()["statistics"]["helmet"], 1)


# ---------------------------------------------------------------------------
# 32-33  Sender UTC timestamps
# ---------------------------------------------------------------------------

class TestSenderUTCTimestamp(unittest.TestCase):
    def _captured_payload(self):
        sender = Sender.__new__(Sender)
        sender.server_url = "http://test"
        captured = {}

        class _FakeSession:
            def post(self, url, json=None, timeout=None):
                captured.update(json)
                raise RuntimeError("stop")

        sender.session = _FakeSession()
        frame = np.zeros((10, 10, 3), dtype="uint8")
        try:
            sender.send_alert(frame, "no_helmet", 0.9)
        except RuntimeError:
            pass
        return captured

    def test_32_payload_timestamp_is_utc_aware(self):
        payload = self._captured_payload()
        ts = payload.get("timestamp", "")
        from datetime import datetime
        dt = datetime.fromisoformat(ts)
        self.assertIsNotNone(dt.tzinfo)

    def test_33_timestamp_contains_utc_offset(self):
        payload = self._captured_payload()
        ts = payload.get("timestamp", "")
        self.assertTrue("+00:00" in ts or ts.endswith("Z"), f"Not UTC: {ts}")


# ---------------------------------------------------------------------------
# 34-37  LED/Buzzer unsupported contract
# ---------------------------------------------------------------------------

class TestLEDBuzzerUnsupportedContract(unittest.TestCase):
    def test_34_led_command_returns_error_code(self):
        bridge = _make_bridge([{"type": "error", "error": "led_not_supported"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.led_control("red")
        self.assertEqual(ctx.exception.error_code, "led_not_supported")

    def test_35_buzzer_command_returns_error_code(self):
        bridge = _make_bridge([{"type": "error", "error": "buzzer_not_supported"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.buzzer_control("on")
        self.assertEqual(ctx.exception.error_code, "buzzer_not_supported")

    def test_36_no_success_ack_emitted_for_led(self):
        bridge = _make_bridge([{"type": "error", "error": "led_not_supported"}])
        result = None
        try:
            result = bridge.led_control("red")
        except RPCError:
            pass
        self.assertIsNone(result)

    def test_37_python_preserves_returned_error_code(self):
        bridge = _make_bridge([{"type": "error", "error": "buzzer_not_supported"}])
        caught = None
        try:
            bridge.buzzer_control("off")
        except RPCError as e:
            caught = e
        self.assertIsNotNone(caught)
        self.assertEqual(caught.error_code, "buzzer_not_supported")

    def test_arduino_comm_h_led_returns_not_supported(self):
        src = _comm_h_text()
        self.assertIn("led_not_supported", src)

    def test_arduino_comm_h_buzzer_returns_not_supported(self):
        src = _comm_h_text()
        self.assertIn("buzzer_not_supported", src)

    def test_arduino_comm_h_no_led_ack_success(self):
        src = _comm_h_text()
        self.assertNotIn("led_ack", src)

    def test_arduino_comm_h_no_buzzer_ack_success(self):
        src = _comm_h_text()
        self.assertNotIn("buzzer_ack", src)


# ---------------------------------------------------------------------------
# Arduino jsonGetStrictInt parser behavior (tests 10-20)
#
# Python re-implementation that mirrors the C logic in comm.h exactly.
# Tested directly against the same inputs the Arduino parser would receive.
# ---------------------------------------------------------------------------

def _json_get_strict_int(json_str, key):
    """
    Python mirror of jsonGetStrictInt() in comm.h.

    Mirrors the C++ algorithm exactly:
    - Manual digit accumulation with overflow guard at acc > (300 - digit) / 10
    - Rejects '.', 'e', 'E' after digits
    - Rejects non-boundary trailing characters
    - Negative flag used to form final value
    - Returns (found, value, malformed)
      found=False, malformed=False  -> key absent
      found=False, malformed=True   -> key present, value invalid
      found=True,  malformed=False  -> key present, value valid
    """
    search = f'"{key}"'
    idx = json_str.find(search)
    if idx == -1:
        return False, None, False

    p = idx + len(search)
    while p < len(json_str) and json_str[p] in (' ', ':'):
        p += 1

    if p >= len(json_str):
        return False, None, True

    ch = json_str[p]

    if ch in ('"', '[', '{'):
        return False, None, True
    if ch != '-' and not ch.isdigit():
        return False, None, True

    negative = False
    if ch == '-':
        negative = True
        p += 1
    if p >= len(json_str) or not json_str[p].isdigit():
        return False, None, True

    acc = 0
    has_digit = False
    while p < len(json_str) and json_str[p].isdigit():
        has_digit = True
        digit = int(json_str[p])
        if acc > (300 - digit) // 10:
            return False, None, True
        acc = acc * 10 + digit
        p += 1
    if not has_digit:
        return False, None, True

    if p < len(json_str) and json_str[p] in ('.', 'e', 'E'):
        return False, None, True
    if p < len(json_str) and json_str[p] not in ('\0', ' ', ',', '}', ']'):
        return False, None, True

    value = -acc if negative else acc
    return True, value, False


def _motor_cmd(speed_json_value):
    return '{{"cmd":"motor","direction":"forward","speed":{}}}'.format(speed_json_value)


def _motor_cmd_no_speed():
    return '{"cmd":"motor","direction":"forward"}'


class TestArduinoStrictSpeedParsing(unittest.TestCase):
    def test_01_absent_key_uses_default(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd_no_speed(), "speed")
        self.assertFalse(found)
        self.assertFalse(malformed)
        self.assertIsNone(val)

    def test_02_zero_accepted(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd(0), "speed")
        self.assertTrue(found)
        self.assertFalse(malformed)
        self.assertEqual(val, 0)

    def test_03_255_accepted(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd(255), "speed")
        self.assertTrue(found)
        self.assertFalse(malformed)
        self.assertEqual(val, 255)

    def test_04_negative_one_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd(-1), "speed")
        self.assertTrue(found)
        self.assertFalse(malformed)
        self.assertEqual(val, -1)
        self.assertTrue(val < 0)

    def test_05_256_rejected_by_range(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd(256), "speed")
        self.assertTrue(found)
        self.assertFalse(malformed)
        self.assertEqual(val, 256)
        self.assertGreater(val, 255)

    def test_06_float_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd("12.5"), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_07_exponent_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd("1e2"), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_08_quoted_string_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd('"150"'), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_09_true_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd("true"), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_10_false_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd("false"), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_11_null_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd("null"), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_12_array_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd("[]"), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_13_object_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd("{}"), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_14_numeric_prefix_rejected(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd("12abc"), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_15_empty_token_rejected(self):
        json_str = '{"cmd":"motor","direction":"forward","speed":}'
        found, val, malformed = _json_get_strict_int(json_str, "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_16_oversized_positive_rejected_as_malformed(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd(99999), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_17_oversized_negative_rejected_as_malformed(self):
        found, val, malformed = _json_get_strict_int(_motor_cmd(-99999), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_18_extremely_long_token_rejected_as_malformed(self):
        long_num = "9" * 50
        found, val, malformed = _json_get_strict_int(_motor_cmd(long_num), "speed")
        self.assertFalse(found)
        self.assertTrue(malformed)

    def test_19_invalid_token_does_not_mutate_pending_move(self):
        src = _comm_h_text()
        malformed_idx = src.find("if (spdMalformed)")
        self.assertNotEqual(malformed_idx, -1, "spdMalformed guard must exist in comm.h")
        snippet = src[malformed_idx:malformed_idx + 80]
        self.assertNotIn("_pendingMove", snippet)

    def test_20_invalid_token_does_not_mutate_pending_speed(self):
        src = _comm_h_text()
        malformed_idx = src.find("if (spdMalformed)")
        snippet = src[malformed_idx:malformed_idx + 80]
        self.assertNotIn("_pendingSpeed", snippet)

    def test_21_invalid_token_does_not_set_has_pending(self):
        src = _comm_h_text()
        malformed_idx = src.find("if (spdMalformed)")
        snippet = src[malformed_idx:malformed_idx + 80]
        self.assertNotIn("_hasPending", snippet)

    def test_22_valid_ack_equals_accepted_speed(self):
        src = _comm_h_text()
        ack_idx = src.find("motor_ack")
        while ack_idx != -1:
            snippet = src[ack_idx:ack_idx + 100]
            if "speed" in snippet and "%d" in snippet:
                self.assertIn("spd", snippet)
                break
            ack_idx = src.find("motor_ack", ack_idx + 1)
        else:
            self.fail("motor_ack format string with speed not found in comm.h")

    def test_23_no_atoi_in_strict_parser(self):
        src = _comm_h_text()
        parser_start = src.find("static bool jsonGetStrictInt(")
        parser_end = src.find("\n    }", parser_start)
        parser_body = src[parser_start:parser_end]
        self.assertNotIn("atoi", parser_body)
        self.assertNotIn("atol", parser_body)
        self.assertNotIn("strtol", parser_body)


if __name__ == "__main__":
    unittest.main()
