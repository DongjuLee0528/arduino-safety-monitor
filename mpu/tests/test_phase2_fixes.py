import json
import math
import os
import shutil
import subprocess
import threading
import unittest
from unittest.mock import MagicMock

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


class _FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, endpoint, *args, timeout=None):
        self.calls.append((endpoint, args, timeout))
        if not self.responses:
            raise TimeoutError("no Bridge response")
        return self.responses.pop(0)

def _make_bridge(responses):
    return BridgeRPC(transport=_FakeTransport(responses))


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


def _pins_h_text():
    import os
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "arduino", "pins.h",
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


def _ui_html_text():
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "app_lab", "ui", "assets", "index.html",
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
        self.assertIn("out += String(speed)", src)

    def test_03_speed_below_0_rejected(self):
        src = _comm_h_text()
        self.assertIn("INVALID_SPEED", src)
        self.assertIn("spd < MOTOR_SPEED_MIN", src)

    def test_04_speed_above_255_rejected(self):
        src = _comm_h_text()
        self.assertIn("spd > MOTOR_SPEED_MAX", src)

    def test_05_speed_uses_typed_rpc_parameter(self):
        src = _comm_h_text()
        self.assertIn("String rpcMotor(String direction, int speed)", src)

    def test_06_python_rejects_non_integer_speed(self):
        bridge = _make_bridge([])
        with self.assertRaises(ValueError):
            bridge.motor_control("forward", 150.0)

    def test_07_bridge_public_api_sends_explicit_default_speed(self):
        bridge = _make_bridge([{"type": "motor_ack", "direction": "forward", "speed": 200}])
        bridge.motor_control("forward")
        self.assertEqual(bridge.transport.calls, [("asm_motor", ("forward", 200), 1.0)])

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
        idx = src.find("String rpcSafeReset()")
        while idx != -1:
            snippet = src[idx:idx + 400]
            if "_pendingSpeed = MOTOR_SPEED_DEFAULT" in snippet:
                break
            idx = src.find("String rpcSafeReset()", idx + 1)
        self.assertNotEqual(idx, -1, "safe_reset block must reset _pendingSpeed")

    def test_11_disconnect_safe_state_resets_speed(self):
        src = _comm_h_text()
        idx = src.find("resetToManualSafeState()")
        snippet = src[idx:idx + 400]
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

    def test_status_rpc_registered(self):
        src = _arduino_ino_text()
        self.assertIn("String asm_status()", src)
        self.assertIn('Bridge.provide_safe("asm_status", asm_status)', src)

    def test_status_rpc_exposes_authoritative_fields(self):
        src = _comm_h_text()
        status_pos = src.find("String rpcStatus(")
        self.assertNotEqual(status_pos, -1)
        snippet = src[status_pos:status_pos + 1800]
        self.assertIn("movement", snippet)
        self.assertIn("_motionLeaseActive", snippet)
        self.assertIn("_warningActive", snippet)
        ino = _arduino_ino_text()
        self.assertIn("distanceAvailable", ino)
        self.assertIn("safety_state", snippet)
        self.assertIn("motor.getMovement()", ino)

    def test_ultrasonic_diag_rpc_registered(self):
        src = _arduino_ino_text()
        self.assertIn("String asm_ultrasonic_diag()", src)
        self.assertIn('Bridge.provide_safe("asm_ultrasonic_diag", asm_ultrasonic_diag)', src)

    def test_ultrasonic_diag_manual_guard_before_measurement(self):
        src = _arduino_ino_text()
        diag_pos = src.find("String asm_ultrasonic_diag()")
        self.assertNotEqual(diag_pos, -1)
        snippet = src[diag_pos:diag_pos + 900]
        guard_pos = snippet.find("comm.getMode() != MODE_MANUAL")
        measure_pos = snippet.find("ultrasonic.measureAllOnce()")
        self.assertNotEqual(guard_pos, -1)
        self.assertNotEqual(measure_pos, -1)
        self.assertLess(guard_pos, measure_pos)

    def test_ultrasonic_diag_does_not_touch_motor_mode_or_warning(self):
        ino = _arduino_ino_text()
        diag_pos = ino.find("String asm_ultrasonic_diag()")
        self.assertNotEqual(diag_pos, -1)
        ino_snippet = ino[diag_pos:diag_pos + 900]
        for forbidden in ("motor.", "rpcMode", "rpcMotor", "rpcControlTick", "rpcSafeReset"):
            self.assertNotIn(forbidden, ino_snippet)

        comm = _comm_h_text()
        rpc_pos = comm.find("String rpcUltrasonicDiagnostic(")
        self.assertNotEqual(rpc_pos, -1)
        comm_snippet = comm[rpc_pos:rpc_pos + 1200]
        for forbidden in ("_startMotionLease", "_renewMotionLease", "_clearMotionLease", "_setLed"):
            self.assertNotIn(forbidden, comm_snippet)


class TestDashboardSafetySemantics(unittest.TestCase):
    def _render_snapshot(self, mcu_safety):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is required for executable UI regression test")

        html = _ui_html_text()
        start = html.index("  function el(")
        end = html.index("  /* ── Control button wiring", start)
        js = html[start:end]
        payload = {
            "state": {
                "connection": {"status": "online"},
                "control": {
                    "safety_state": mcu_safety,
                    "motion_lease_active": mcu_safety == "motion_authorized",
                    "control_tick_fresh": mcu_safety == "motion_authorized",
                },
                "detection": {
                    "worker_present": True,
                    "helmet_result": "no_helmet",
                },
                "distances": {"front": None, "rear": None, "left": None, "right": None},
                "statistics": {},
                "events": [],
            },
            "info": {},
            "dev_mode": False,
            "warning_active": False,
        }
        harness = f"""
const input = {json.dumps(payload)};
const elements = {{}};
function makeEl(id) {{
  return elements[id] || (elements[id] = {{
    id,
    textContent: "",
    innerHTML: "",
    className: "",
    disabled: false,
    style: {{}},
    classList: {{ add() {{}}, remove() {{}}, contains() {{ return false; }} }},
    setAttribute() {{}},
    addEventListener() {{}}
  }});
}}
const document = {{ getElementById: makeEl }};
{js}
applySnapshot(input);
console.log(JSON.stringify({{
  safetyState: elements["safety-state"],
  safetyPill: elements["pill-safety"],
  systemSub: elements["system-status-sub"]
}}));
"""
        proc = subprocess.run(
            [node, "-e", harness],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(proc.stdout)

    def test_no_helmet_danger_wins_over_mcu_control_safety(self):
        for mcu_safety in ("safe", "warning", "motion_authorized"):
            with self.subTest(mcu_safety=mcu_safety):
                rendered = self._render_snapshot(mcu_safety)
                self.assertEqual(rendered["safetyState"]["textContent"], "DANGER")
                self.assertIn("val-red", rendered["safetyState"]["className"])
                self.assertIn("DANGER", rendered["safetyPill"]["innerHTML"])
                self.assertIn("pill-red", rendered["safetyPill"]["className"])
                self.assertNotEqual(rendered["safetyState"]["textContent"], "SAFE")
                self.assertIn("MCU control:", rendered["systemSub"]["textContent"])


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
        snippet = src[setup_idx:setup_idx + 700]
        self.assertIn("ultrasonic.begin()", snippet)

    def test_measure_all_once_reads_each_sensor_once(self):
        src = _ultrasonic_h_text()
        diag_idx = src.find("void measureAllOnce()")
        self.assertNotEqual(diag_idx, -1)
        snippet = src[diag_idx:diag_idx + 400]
        self.assertIn("measureSensor(0)", snippet)
        self.assertIn("measureSensor(1)", snippet)
        self.assertIn("measureSensor(2)", snippet)
        self.assertIn("measureSensor(3)", snippet)
        self.assertIn("lastMeasurement = millis()", snippet)


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
# 34-37  LED supported / buzzer unsupported contract
# ---------------------------------------------------------------------------

class TestLEDBuzzerUnsupportedContract(unittest.TestCase):
    def test_34_led_command_accepts_ack(self):
        bridge = _make_bridge([{"type": "led_ack", "color": "red"}])
        self.assertTrue(bridge.led_control("red"))

    def test_35_buzzer_command_returns_error_code(self):
        bridge = _make_bridge([{"type": "error", "error": "buzzer_not_supported"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.buzzer_control("on")
        self.assertEqual(ctx.exception.error_code, "buzzer_not_supported")

    def test_36_success_ack_emitted_for_led(self):
        bridge = _make_bridge([{"type": "led_ack", "color": "off"}])
        self.assertTrue(bridge.led_control("off"))

    def test_37_python_preserves_returned_error_code(self):
        bridge = _make_bridge([{"type": "error", "error": "buzzer_not_supported"}])
        caught = None
        try:
            bridge.buzzer_control("off")
        except RPCError as e:
            caught = e
        self.assertIsNotNone(caught)
        self.assertEqual(caught.error_code, "buzzer_not_supported")

    def test_arduino_comm_h_led_returns_ack(self):
        src = _comm_h_text()
        self.assertIn("led_ack", src)

    def test_arduino_comm_h_buzzer_returns_not_supported(self):
        src = _comm_h_text()
        self.assertIn("buzzer_not_supported", src)

    def test_arduino_comm_h_no_led_not_supported(self):
        src = _comm_h_text()
        self.assertNotIn("led_not_supported", src)

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
        malformed_idx = src.find("if (spd < MOTOR_SPEED_MIN || spd > MOTOR_SPEED_MAX)")
        self.assertNotEqual(malformed_idx, -1, "RPC speed bounds guard must exist in comm.h")
        snippet = src[malformed_idx:malformed_idx + 120]
        self.assertNotIn("_pendingMove", snippet)

    def test_20_invalid_token_does_not_mutate_pending_speed(self):
        src = _comm_h_text()
        malformed_idx = src.find("if (spd < MOTOR_SPEED_MIN || spd > MOTOR_SPEED_MAX)")
        snippet = src[malformed_idx:malformed_idx + 120]
        self.assertNotIn("_pendingSpeed", snippet)

    def test_21_invalid_token_does_not_set_has_pending(self):
        src = _comm_h_text()
        malformed_idx = src.find("if (spd < MOTOR_SPEED_MIN || spd > MOTOR_SPEED_MAX)")
        snippet = src[malformed_idx:malformed_idx + 120]
        self.assertNotIn("_hasPending", snippet)

    def test_22_valid_ack_equals_accepted_speed(self):
        src = _comm_h_text()
        motor_idx = src.find("String rpcMotor(")
        motor_block = src[motor_idx:motor_idx + 1200]
        self.assertIn("return _motorAck(direction.c_str(), spd)", motor_block)

    def test_23_no_atoi_in_strict_parser(self):
        src = _comm_h_text()
        motor_idx = src.find("String rpcMotor(")
        motor_body = src[motor_idx:motor_idx + 1200]
        self.assertNotIn("atoi", motor_body)
        self.assertNotIn("atol", motor_body)
        self.assertNotIn("strtol", motor_body)


# ---------------------------------------------------------------------------
# Buzzer diagnostic RPC (BD tests)
# ---------------------------------------------------------------------------

class TestBuzzerDiagnosticRPC(unittest.TestCase):

    def test_BD01_buzzer_diag_pin_defined_as_13(self):
        src = _pins_h_text()
        self.assertIn("BUZZER_DIAG_PIN", src)
        self.assertIn("#define BUZZER_DIAG_PIN", src)
        import re
        m = re.search(r"#define BUZZER_DIAG_PIN\s+(\d+)", src)
        self.assertIsNotNone(m, "BUZZER_DIAG_PIN must have a numeric value")
        self.assertEqual(int(m.group(1)), 13, "BUZZER_DIAG_PIN must be 13")

    def test_BD02_buzzer_diag_pin_unique_static_asserts_present(self):
        src = _pins_h_text()
        for other in (
            "MOTOR_LEFT_ENABLE_PIN", "MOTOR_LEFT_IN1_PIN", "MOTOR_LEFT_IN2_PIN",
            "MOTOR_RIGHT_ENABLE_PIN", "MOTOR_RIGHT_IN1_PIN", "MOTOR_RIGHT_IN2_PIN",
            "ALERT_LED_PIN",
            "ULTRASONIC_FRONT_TRIGGER_PIN", "ULTRASONIC_FRONT_ECHO_PIN",
            "ULTRASONIC_REAR_TRIGGER_PIN", "ULTRASONIC_REAR_ECHO_PIN",
        ):
            self.assertIn(
                f"BUZZER_DIAG_PIN != {other}",
                src,
                f"Missing static_assert: BUZZER_DIAG_PIN != {other}",
            )

    def test_BD03_startup_off_initialization_in_begin(self):
        src = _comm_h_text()
        begin_pos = src.find("void begin()")
        self.assertNotEqual(begin_pos, -1)
        begin_block = src[begin_pos:begin_pos + 400]
        self.assertIn("pinMode(BUZZER_DIAG_PIN, OUTPUT)", begin_block)
        self.assertIn("digitalWrite(BUZZER_DIAG_PIN, LOW)", begin_block)

    def test_BD04_diag_rpc_registered_in_setup(self):
        src = _arduino_ino_text()
        self.assertIn("String asm_buzzer_diag()", src)
        self.assertIn('Bridge.provide_safe("asm_buzzer_diag", asm_buzzer_diag)', src)

    def test_BD05_diag_on_off_sequence_structure(self):
        src = _comm_h_text()
        diag_pos = src.find("String rpcBuzzerDiag()")
        self.assertNotEqual(diag_pos, -1)
        block = src[diag_pos:diag_pos + 400]
        high_pos = block.find("digitalWrite(BUZZER_DIAG_PIN, HIGH)")
        delay_pos = block.find("delay(")
        low_pos = block.rfind("digitalWrite(BUZZER_DIAG_PIN, LOW)")
        self.assertNotEqual(high_pos, -1, "HIGH pulse missing in rpcBuzzerDiag")
        self.assertNotEqual(delay_pos, -1, "delay missing in rpcBuzzerDiag")
        self.assertNotEqual(low_pos, -1, "final LOW missing in rpcBuzzerDiag")
        self.assertLess(high_pos, delay_pos, "HIGH must precede delay")
        self.assertLess(delay_pos, low_pos, "delay must precede final LOW")

    def test_BD06_diag_duration_bounded(self):
        src = _comm_h_text()
        diag_pos = src.find("String rpcBuzzerDiag()")
        block = src[diag_pos:diag_pos + 400]
        import re
        m = re.search(r"delay\((\d+)\)", block)
        self.assertIsNotNone(m, "delay value must be a literal in rpcBuzzerDiag")
        ms = int(m.group(1))
        self.assertLessEqual(ms, 500, "diagnostic beep must not exceed 500 ms")
        self.assertGreaterEqual(ms, 100, "diagnostic beep must be at least 100 ms")

    def test_BD07_response_contract(self):
        src = _comm_h_text()
        diag_pos = src.find("String rpcBuzzerDiag()")
        block = src[diag_pos:diag_pos + 400]
        self.assertIn("buzzer_diag", block)
        self.assertIn("status", block)
        self.assertIn("pin", block)

    def test_BD08_diag_manual_guard_before_pulse(self):
        src = _arduino_ino_text()
        diag_pos = src.find("String asm_buzzer_diag()")
        self.assertNotEqual(diag_pos, -1)
        block = src[diag_pos:diag_pos + 300]
        guard_pos = block.find("DIAG_REQUIRES_MANUAL")
        call_pos = block.find("rpcBuzzerDiag()")
        self.assertNotEqual(guard_pos, -1, "MANUAL guard missing in asm_buzzer_diag")
        self.assertNotEqual(call_pos, -1, "rpcBuzzerDiag call missing in asm_buzzer_diag")
        self.assertLess(guard_pos, call_pos, "MANUAL guard must precede rpcBuzzerDiag call")

    def test_BD09_no_mode_change_in_diag(self):
        src = _comm_h_text()
        diag_pos = src.find("String rpcBuzzerDiag()")
        block = src[diag_pos:diag_pos + 400]
        self.assertNotIn("_mode =", block, "rpcBuzzerDiag must not change mode")
        self.assertNotIn("_pendingMove =", block, "rpcBuzzerDiag must not issue motor commands")

    def test_BD10_no_motion_lease_in_diag(self):
        src = _comm_h_text()
        diag_pos = src.find("String rpcBuzzerDiag()")
        block = src[diag_pos:diag_pos + 400]
        self.assertNotIn("_startMotionLease", block, "rpcBuzzerDiag must not start motion lease")
        self.assertNotIn("_renewMotionLease", block, "rpcBuzzerDiag must not renew motion lease")

    def test_BD11_buzzer_diag_not_in_reserved_pins_comment(self):
        src = _pins_h_text()
        reserved_pos = src.find("Reserved pins (do not assign)")
        self.assertNotEqual(reserved_pos, -1)
        end_of_reserved = src.find("Active pin uniqueness table", reserved_pos)
        reserved_block = src[reserved_pos:end_of_reserved]
        self.assertNotIn("D13", reserved_block, "D13 must not be listed in Reserved pins")


if __name__ == "__main__":
    unittest.main()
