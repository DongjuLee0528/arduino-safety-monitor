import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mpu.dashboard_state import DashboardState
from mpu.main import HelmetDetectionSystem, _EventSuppressor


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


def _person(bbox):
    return {"bbox": bbox, "confidence": 0.9}


def _comm_h_text():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "arduino", "comm.h")
    )
    with open(path) as f:
        return f.read()


def _robot_controller_h_text():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "arduino", "robot_controller.h")
    )
    with open(path) as f:
        return f.read()


def _config_h_text():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "arduino", "config.h")
    )
    with open(path) as f:
        return f.read()


class TestAcceptedWorkerHelmetWarning(unittest.TestCase):
    def test_new_no_helmet_worker_starts_warning(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.process_frame(_frame())

        system.bridge_rpc.motor_control.assert_not_called()
        system.bridge_rpc.led_control.assert_called_once_with("red")

    def test_same_worker_does_not_restart_warning(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.process_frame(_frame())
        system.process_frame(_frame())
        system.process_frame(_frame())

        system.bridge_rpc.motor_control.assert_not_called()
        system.bridge_rpc.led_control.assert_called_once_with("red")

    def test_second_worker_starts_second_warning(self):
        system = _make_system()
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [_person([300, 0, 100, 200])]
        system.process_frame(_frame())

        system.bridge_rpc.motor_control.assert_not_called()
        self.assertEqual(system.bridge_rpc.led_control.call_count, 2)
        system.bridge_rpc.led_control.assert_any_call("red")

    def test_helmet_worker_does_not_start_warning(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}

        system.process_frame(_frame())

        system.bridge_rpc.motor_control.assert_not_called()
        system.bridge_rpc.led_control.assert_not_called()

    def test_motion_lease_tick_still_runs_in_main_loop(self):
        system = _make_system()
        system._last_tick_time = 0.0
        system.bridge_rpc.connect.return_value = None
        system.camera.capture_frame.side_effect = RuntimeError("stop loop")

        with patch("mpu.main.time.monotonic", return_value=1.0):
            system.start()

        system.bridge_rpc.control_tick.assert_called_once()

    def test_heartbeat_still_uses_ping_not_control_tick(self):
        import inspect
        from mpu.bridge_rpc import BridgeRPC

        src = inspect.getsource(BridgeRPC._heartbeat_loop)
        self.assertIn('"ping"', src)
        self.assertNotIn("control_tick", src)


class _WarningSim:
    def __init__(self):
        self.now = 0
        self.warning_active = False
        self.warning_start = 0
        self.last_led_toggle = 0
        self.led_on = False
        self.mode = "manual"
        self.pending_move = "none"
        self.has_pending = False
        self.stop_latched = False
        self.motion_lease_active = False
        self.last_motion_lease = 0

    def advance(self, ms):
        self.now += ms
        self.update_warning()

    def process(self, payload):
        cmd = json.loads(payload)
        if cmd["cmd"] == "ping":
            return {"type": "pong"}
        if cmd["cmd"] == "control_tick":
            if self.motion_lease_active:
                self.last_motion_lease = self.now
                return {"type": "control_tick_ack", "motion_authorized": True}
            return {"type": "control_tick_ack", "motion_authorized": False}
        if cmd["cmd"] == "mode":
            if cmd["value"] == "auto" and not self.stop_latched and not self.warning_active:
                self.mode = "auto"
                self.motion_lease_active = True
                self.last_motion_lease = self.now
            elif cmd["value"] == "manual":
                self.mode = "manual"
                self.motion_lease_active = False
            return {"type": "mode_ack", "mode": cmd["value"]}
        if cmd["cmd"] == "motor":
            direction = cmd["direction"]
            if direction == "stop":
                self.mode = "manual"
                self.pending_move = "stop"
                self.has_pending = True
                self.stop_latched = True
                self.motion_lease_active = False
            elif not self.stop_latched and not self.warning_active:
                self.pending_move = direction
                self.has_pending = True
                self.motion_lease_active = True
                self.last_motion_lease = self.now
            return {"type": "motor_ack", "direction": direction, "speed": cmd.get("speed", 150)}
        if cmd["cmd"] == "led":
            if cmd["value"] == "red" and not self.warning_active:
                self.warning_active = True
                self.warning_start = self.now
                self.last_led_toggle = self.now
                self.mode = "manual"
                self.pending_move = "stop"
                self.has_pending = True
                self.stop_latched = True
                self.led_on = True
            elif cmd["value"] == "off" and not self.warning_active:
                self.led_on = False
            return {"type": "led_ack", "color": cmd["value"]}
        return {"type": "error", "error": "UNKNOWN_CMD"}

    def consume_pending(self):
        cmd = self.pending_move
        self.pending_move = "none"
        self.has_pending = False
        self.stop_latched = False
        return cmd

    def update_warning(self):
        if not self.warning_active:
            return
        if self.now - self.warning_start >= 10000:
            self.warning_active = False
            self.led_on = False
            self.mode = "manual"
            self.pending_move = "none"
            self.has_pending = False
            self.stop_latched = False
            return
        if self.now - self.last_led_toggle >= 250:
            self.last_led_toggle = self.now
            self.led_on = not self.led_on


class TestMcuWarningBehavior(unittest.TestCase):
    def test_warning_starts_with_stop_and_led_on(self):
        sim = _WarningSim()
        resp = sim.process('{"cmd":"led","value":"red"}')
        self.assertEqual(resp["type"], "led_ack")
        self.assertTrue(sim.warning_active)
        self.assertEqual(sim.pending_move, "stop")
        self.assertTrue(sim.led_on)

    def test_blink_timing_250_ms_on_off(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        self.assertTrue(sim.led_on)
        sim.advance(249)
        self.assertTrue(sim.led_on)
        sim.advance(1)
        self.assertFalse(sim.led_on)
        sim.advance(250)
        self.assertTrue(sim.led_on)

    def test_same_led_red_does_not_restart_timer(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(5000)
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(4999)
        self.assertTrue(sim.warning_active)
        sim.advance(1)
        self.assertFalse(sim.warning_active)

    def test_warning_expiry_led_off_stopped_manual_no_auto_restore(self):
        sim = _WarningSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.process('{"cmd":"led","value":"red"}')
        sim.consume_pending()
        sim.advance(10000)

        self.assertFalse(sim.warning_active)
        self.assertFalse(sim.led_on)
        self.assertEqual(sim.mode, "manual")
        self.assertFalse(sim.has_pending)
        self.assertEqual(sim.pending_move, "none")

    def test_motion_command_during_warning_does_not_execute_or_queue(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.consume_pending()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertFalse(sim.has_pending)
        self.assertEqual(sim.pending_move, "none")

    def test_control_tick_continues_during_warning(self):
        sim = _WarningSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(250)
        resp = sim.process('{"cmd":"control_tick"}')
        self.assertEqual(resp, {"type": "control_tick_ack", "motion_authorized": True})

    def test_ping_continues_during_warning(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        self.assertEqual(sim.process('{"cmd":"ping"}'), {"type": "pong"})


class TestWarningSourceStructure(unittest.TestCase):
    def test_warning_duration_defined_as_10000(self):
        self.assertIn("#define WARNING_DURATION_MS    10000", _config_h_text())

    def test_blink_interval_defined_as_250(self):
        self.assertIn("#define LED_BLINK_INTERVAL_MS  250", _config_h_text())

    def test_comm_uses_millis_not_delay_for_warning(self):
        src = _comm_h_text()
        warning_pos = src.find("void updateWarning()")
        self.assertNotEqual(warning_pos, -1)
        warning_block = src[warning_pos:warning_pos + 700]
        self.assertIn("millis()", warning_block)
        self.assertNotIn("delay(", warning_block)
        self.assertNotIn("delayMicroseconds(", warning_block)

    def test_led_not_supported_removed(self):
        self.assertNotIn("led_not_supported", _comm_h_text())

    def test_robot_controller_blocks_motion_but_updates_ultrasonic(self):
        src = _robot_controller_h_text()
        warning_pos = src.find("isWarningActive()")
        self.assertNotEqual(warning_pos, -1)
        warning_block = src[warning_pos:warning_pos + 180]
        self.assertIn("_ultrasonic->update()", warning_block)
        self.assertIn("_motor->stop()", warning_block)
        self.assertIn("return", warning_block)


if __name__ == "__main__":
    unittest.main()
