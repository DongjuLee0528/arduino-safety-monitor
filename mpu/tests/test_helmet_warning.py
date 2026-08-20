import json
import os
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from mpu.alert_manager import AlertManager
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
    system._warning_hardware_active = False
    system._last_violation_detected = False
    system._connected = False
    system.dashboard = DashboardState()
    system._events = _EventSuppressor(system.dashboard, suppress_seconds=0.0)
    system._prev_bboxes = []
    return system


def _make_system_with_real_alert_manager():
    system = _make_system()
    system.alert_manager = AlertManager(callback=system.on_no_helmet_alert)
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
    def test_new_no_helmet_worker_does_not_start_warning_before_confirmation(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.process_frame(_frame())

        system.bridge_rpc.motor_control.assert_not_called()
        system.bridge_rpc.led_control.assert_not_called()
        system.alert_manager.on_detection.assert_called_once_with(True)

    def test_new_no_helmet_worker_does_not_send_external_alert_before_confirmation(self):
        system = _make_system()
        call_order = []
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        system.bridge_rpc.led_control.side_effect = lambda color: call_order.append(("led", color)) or True
        system.sender.send_alert.side_effect = (
            lambda frame, label, confidence: call_order.append(("sender", label)) or True
        )

        system.process_frame(_frame())

        self.assertEqual(call_order, [])

    def test_sender_failure_before_confirmation_does_not_abort_processing(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        system.sender.send_alert.side_effect = RuntimeError("network unavailable")

        system.process_frame(_frame())

        system.sender.send_alert.assert_not_called()
        system.bridge_rpc.led_control.assert_not_called()
        snap = system.dashboard.snapshot()["statistics"]
        self.assertEqual(snap["inspected"], 1)
        self.assertEqual(snap["no_helmet"], 1)

    def test_same_worker_does_not_restart_warning(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.process_frame(_frame())
        system.process_frame(_frame())
        system.process_frame(_frame())

        system.bridge_rpc.motor_control.assert_not_called()
        system.bridge_rpc.led_control.assert_not_called()

    def test_second_worker_does_not_bypass_confirmation(self):
        system = _make_system()
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [_person([300, 0, 100, 200])]
        system.process_frame(_frame())

        system.bridge_rpc.motor_control.assert_not_called()
        system.bridge_rpc.led_control.assert_not_called()

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
        self.assertIn("self.ping()", src)
        self.assertNotIn("control_tick", src)


class TestAlertManagerMcuWarningIntegration(unittest.TestCase):
    def test_no_helmet_frame_one_and_two_do_not_start_warning(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.process_frame(_frame())
        system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_not_called()
        system.sender.send_alert.assert_not_called()

    def test_no_helmet_third_consecutive_frame_starts_one_warning(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        for _ in range(3):
            system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_called_once_with("red")
        system.sender.send_alert.assert_called_once()

    def test_clean_frame_after_confirmed_warning_clears_once(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())

        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.process_frame(_frame())

        self.assertEqual(system.bridge_rpc.led_control.mock_calls, [call("red"), call("off")])

    def test_mixed_helmet_and_no_helmet_warns_after_threshold(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([300, 0, 100, 200]),
        ]

        for _ in range(3):
            system.helmet_classifier.predict.side_effect = [
                {"label": "helmet", "confidence": 0.95},
                {"label": "no_helmet", "confidence": 0.9},
            ]
            system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_called_once_with("red")

    def test_all_helmet_never_starts_warning(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}

        for _ in range(4):
            system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_not_called()

    def test_zero_persons_never_starts_warning(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = []

        for _ in range(4):
            system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_not_called()

    def test_unknown_classifier_result_never_starts_warning(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "unknown", "confidence": 0.5}

        for _ in range(4):
            system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_not_called()

    def test_camera_failure_clean_transition_clears_confirmed_warning_once(self):
        system = _make_system_with_real_alert_manager()
        system._last_tick_time = 1.0
        system._last_telemetry_time = 1.0
        system.bridge_rpc.connect.return_value = None
        system.camera.capture_frame.side_effect = RuntimeError("camera unavailable")
        system._warning_hardware_active = True
        system._last_violation_detected = True

        with patch("mpu.main.time.monotonic", return_value=1.0):
            system.start()

        system.bridge_rpc.led_control.assert_called_once_with("off")

    def test_confirmed_warning_statistics_and_events_count_once(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        for _ in range(3):
            system.process_frame(_frame())

        snap = system.dashboard.snapshot()["statistics"]
        self.assertEqual(snap["warnings"], 1)
        messages = [event["message"] for event in system.dashboard.snapshot()["events"]]
        self.assertEqual(messages.count("No-helmet alert triggered"), 1)
        self.assertEqual(messages.count("No-helmet warning started"), 1)

    def test_cooldown_still_blocks_immediate_second_warning(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())

        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [_person([300, 0, 100, 200])]
        for _ in range(3):
            system.process_frame(_frame())

        self.assertEqual(system.bridge_rpc.led_control.mock_calls, [call("red"), call("off")])
        system.sender.send_alert.assert_called_once()

    def test_clean_frame_resets_streak_before_external_alert(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        system.process_frame(_frame())
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.process_frame(_frame())
        system.process_frame(_frame())

        system.sender.send_alert.assert_not_called()

    def test_sender_failure_after_confirmation_does_not_skip_warning_or_stats(self):
        system = _make_system_with_real_alert_manager()
        system.sender.send_alert.side_effect = RuntimeError("offline")
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        for _ in range(3):
            system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_called_once_with("red")
        self.assertEqual(system.dashboard.snapshot()["statistics"]["warnings"], 1)
        system.sender.send_alert.assert_called_once()

    def test_mixed_multi_person_confirmed_sends_one_external_alert(self):
        system = _make_system_with_real_alert_manager()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([300, 0, 100, 200]),
        ]

        for _ in range(3):
            system.helmet_classifier.predict.side_effect = [
                {"label": "helmet", "confidence": 0.95},
                {"label": "no_helmet", "confidence": 0.9},
            ]
            system.process_frame(_frame())

        system.sender.send_alert.assert_called_once()


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
        self.manual_hold_active = False
        self.last_manual_hold = 0

    def advance(self, ms):
        self.now += ms
        self.update_warning()

    def process(self, payload):
        cmd = json.loads(payload)
        if cmd["cmd"] == "ping":
            return {"type": "pong"}
        if cmd["cmd"] == "control_tick":
            if self.motion_lease_active and (self.mode == "auto" or self.manual_hold_active):
                self.last_motion_lease = self.now
                return {"type": "control_tick_ack", "motion_authorized": True}
            return {"type": "control_tick_ack", "motion_authorized": False}
        if cmd["cmd"] == "manual_hold":
            if self.warning_active:
                return {"type": "error", "error": "CMD_BLOCKED_BY_WARNING"}
            if self.stop_latched:
                return {"type": "error", "error": "CMD_BLOCKED_BY_STOP_LATCH"}
            if self.mode != "manual" or not self.motion_lease_active or not self.manual_hold_active:
                return {"type": "manual_hold_ack", "active": False}
            self.last_manual_hold = self.now
            self.last_motion_lease = self.now
            return {"type": "manual_hold_ack", "active": True}
        if cmd["cmd"] == "mode":
            if cmd["value"] == "auto" and self.warning_active:
                return {"type": "error", "error": "CMD_BLOCKED_BY_WARNING"}
            if cmd["value"] == "auto" and self.stop_latched:
                return {"type": "error", "error": "CMD_BLOCKED_BY_STOP_LATCH"}
            if cmd["value"] == "auto":
                self.mode = "auto"
                self.motion_lease_active = True
                self.last_motion_lease = self.now
            elif cmd["value"] == "manual":
                self.mode = "manual"
                self.motion_lease_active = False
                self.manual_hold_active = False
            return {"type": "mode_ack", "mode": cmd["value"]}
        if cmd["cmd"] == "motor":
            direction = cmd["direction"]
            if direction == "stop":
                self.mode = "manual"
                self.pending_move = "stop"
                self.has_pending = True
                self.stop_latched = True
                self.motion_lease_active = False
                self.manual_hold_active = False
            elif self.warning_active:
                return {"type": "error", "error": "CMD_BLOCKED_BY_WARNING"}
            elif self.stop_latched:
                return {"type": "error", "error": "CMD_BLOCKED_BY_STOP_LATCH"}
            else:
                self.pending_move = direction
                self.has_pending = True
                self.motion_lease_active = True
                self.last_motion_lease = self.now
                self.manual_hold_active = True
                self.last_manual_hold = self.now
            return {"type": "motor_ack", "direction": direction, "speed": cmd.get("speed", 150)}
        if cmd["cmd"] == "led":
            if cmd["value"] == "red":
                if not self.warning_active:
                    self.warning_active = True
                    self.warning_start = self.now
                    self.last_led_toggle = self.now
                    self.mode = "manual"
                    self.pending_move = "stop"
                    self.has_pending = True
                    self.stop_latched = True
                    self.motion_lease_active = False
                    self.manual_hold_active = False
                    self.led_on = True
                else:
                    self.warning_start = self.now
                    self.last_led_toggle = self.now
                    self.motion_lease_active = False
                    self.manual_hold_active = False
                    self.led_on = True
            elif cmd["value"] == "off":
                self.clear_warning()
            return {"type": "led_ack", "color": cmd["value"]}
        if cmd["cmd"] == "safe_reset":
            self.motion_lease_active = False
            self.manual_hold_active = False
            self.clear_warning()
            self.mode = "manual"
            self.pending_move = "stop"
            self.has_pending = True
            self.stop_latched = True
            return {"type": "safe_reset_ack", "status": "ok", "mode": "manual"}
        return {"type": "error", "error": "UNKNOWN_CMD"}

    def clear_warning(self):
        self.warning_active = False
        self.warning_start = 0
        self.last_led_toggle = 0
        self.led_on = False

    def disconnect(self):
        self.motion_lease_active = False
        self.manual_hold_active = False
        self.clear_warning()

    def reset_to_manual_safe_state(self):
        self.motion_lease_active = False
        self.manual_hold_active = False
        self.clear_warning()
        self.mode = "manual"
        self.pending_move = "none"
        self.has_pending = False
        self.stop_latched = False

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
            self.clear_warning()
            self.motion_lease_active = False
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

    def test_second_led_red_during_active_warning_refreshes_timer(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(5000)
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(9999)
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
        self.assertFalse(sim.motion_lease_active)

    def test_warning_expiry_clears_motion_lease(self):
        sim = _WarningSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        self.assertTrue(sim.motion_lease_active)
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(10000)

        self.assertFalse(sim.motion_lease_active)

    def test_motion_command_during_warning_returns_error_and_does_not_queue(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.consume_pending()
        resp = sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertFalse(sim.has_pending)
        self.assertEqual(sim.pending_move, "none")
        self.assertEqual(resp, {"type": "error", "error": "CMD_BLOCKED_BY_WARNING"})

    def test_auto_mode_during_warning_returns_error(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.consume_pending()
        resp = sim.process('{"cmd":"mode","value":"auto"}')
        self.assertEqual(resp, {"type": "error", "error": "CMD_BLOCKED_BY_WARNING"})
        self.assertEqual(sim.mode, "manual")

    def test_control_tick_does_not_renew_during_warning(self):
        sim = _WarningSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(250)
        resp = sim.process('{"cmd":"control_tick"}')
        self.assertEqual(resp, {"type": "control_tick_ack", "motion_authorized": False})

    def test_ping_continues_during_warning(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        self.assertEqual(sim.process('{"cmd":"ping"}'), {"type": "pong"})

    def test_safe_reset_clears_warning_and_led(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        resp = sim.process('{"cmd":"safe_reset"}')

        self.assertEqual(resp, {"type": "safe_reset_ack", "status": "ok", "mode": "manual"})
        self.assertFalse(sim.warning_active)
        self.assertFalse(sim.led_on)
        self.assertEqual(sim.warning_start, 0)
        self.assertEqual(sim.last_led_toggle, 0)

    def test_disconnect_clears_warning_and_led(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.disconnect()

        self.assertFalse(sim.warning_active)
        self.assertFalse(sim.led_on)

    def test_manual_safe_reset_clears_warning_and_led(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.reset_to_manual_safe_state()

        self.assertFalse(sim.warning_active)
        self.assertFalse(sim.led_on)

    def test_led_off_clears_active_warning_and_led(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        resp = sim.process('{"cmd":"led","value":"off"}')

        self.assertEqual(resp, {"type": "led_ack", "color": "off"})
        self.assertFalse(sim.warning_active)
        self.assertFalse(sim.led_on)


class TestHWV001WarningRefresh(unittest.TestCase):
    def test_first_accepted_worker_starts_warning_at_t0(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        self.assertTrue(sim.warning_active)
        self.assertEqual(sim.warning_start, 0)
        self.assertTrue(sim.led_on)

    def test_same_accepted_worker_does_not_bypass_confirmation(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.process_frame(_frame())
        system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_not_called()

    def test_different_accepted_worker_does_not_bypass_confirmation(self):
        system = _make_system()
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}

        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [_person([300, 0, 100, 200])]
        system.process_frame(_frame())

        system.bridge_rpc.led_control.assert_not_called()

    def test_mcu_refresh_resets_warning_start_time(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(5000)
        sim.process('{"cmd":"led","value":"red"}')
        self.assertEqual(sim.warning_start, 5000)

    def test_mcu_refresh_warning_active_remains_true(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(3000)
        sim.process('{"cmd":"led","value":"red"}')
        self.assertTrue(sim.warning_active)

    def test_mcu_refresh_extends_warning_full_10s_from_refresh(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(5000)
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(9999)
        self.assertTrue(sim.warning_active)
        sim.advance(1)
        self.assertFalse(sim.warning_active)

    def test_mcu_refresh_keeps_mode_manual(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(3000)
        sim.process('{"cmd":"led","value":"red"}')
        self.assertEqual(sim.mode, "manual")

    def test_mcu_refresh_clears_motion_lease(self):
        sim = _WarningSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        self.assertTrue(sim.motion_lease_active)
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(3000)
        sim.process('{"cmd":"led","value":"red"}')
        self.assertFalse(sim.motion_lease_active)

    def test_mcu_refresh_led_immediately_on(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(250)
        self.assertFalse(sim.led_on)
        sim.process('{"cmd":"led","value":"red"}')
        self.assertTrue(sim.led_on)

    def test_mcu_refresh_resets_blink_timestamp(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.advance(5000)
        sim.process('{"cmd":"led","value":"red"}')
        self.assertEqual(sim.last_led_toggle, 5000)

    def test_mcu_refresh_pending_movement_remains_safely_blocked(self):
        sim = _WarningSim()
        sim.process('{"cmd":"led","value":"red"}')
        sim.consume_pending()
        sim.advance(3000)
        sim.process('{"cmd":"led","value":"red"}')
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertFalse(sim.has_pending)
        self.assertEqual(sim.pending_move, "none")

    def test_mcu_refresh_no_delay_in_source(self):
        src = _comm_h_text()
        refresh_pos = src.find("void _refreshWarning()")
        self.assertNotEqual(refresh_pos, -1)
        refresh_block = src[refresh_pos:refresh_pos + 300]
        self.assertNotIn("delay(", refresh_block)
        self.assertNotIn("delayMicroseconds(", refresh_block)

    def test_mcu_refresh_source_sets_led_on(self):
        src = _comm_h_text()
        refresh_pos = src.find("void _refreshWarning()")
        self.assertNotEqual(refresh_pos, -1)
        refresh_block = src[refresh_pos:refresh_pos + 300]
        self.assertIn("_setLed(true)", refresh_block)

    def test_mcu_refresh_source_resets_warning_start_time(self):
        src = _comm_h_text()
        refresh_pos = src.find("void _refreshWarning()")
        self.assertNotEqual(refresh_pos, -1)
        refresh_block = src[refresh_pos:refresh_pos + 300]
        self.assertIn("_warningStartTime", refresh_block)

    def test_mcu_led_red_during_active_warning_calls_refresh_not_start(self):
        src = _comm_h_text()
        led_handler_pos = src.find('value == "red"')
        self.assertNotEqual(led_handler_pos, -1)
        led_block = src[led_handler_pos:led_handler_pos + 300]
        self.assertIn("_refreshWarning()", led_block)
        self.assertIn("_startWarning()", led_block)


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

    def test_led_off_rpc_clears_active_warning_in_source(self):
        src = _comm_h_text()
        off_pos = src.find('value == "off"')
        self.assertNotEqual(off_pos, -1)
        off_block = src[off_pos:off_pos + 120]
        self.assertIn("_clearWarningLed()", off_block)
        self.assertNotIn("!_warningActive", off_block)

    def test_warning_expiry_clears_motion_lease_in_source(self):
        src = _comm_h_text()
        warning_pos = src.find("void updateWarning()")
        self.assertNotEqual(warning_pos, -1)
        warning_block = src[warning_pos:warning_pos + 800]
        expiry_pos = warning_block.find("WARNING_DURATION_MS")
        self.assertNotEqual(expiry_pos, -1)
        expiry_block = warning_block[expiry_pos:expiry_pos + 250]
        self.assertIn("_clearWarningLed()", expiry_block)
        self.assertIn("_clearMotionLease()", expiry_block)

    def test_reset_paths_clear_warning_in_source(self):
        src = _comm_h_text()
        safe_reset_pos = src.find("String rpcSafeReset()")
        self.assertNotEqual(safe_reset_pos, -1)
        self.assertIn("_clearWarningLed()", src[safe_reset_pos:safe_reset_pos + 400])

        disconnect_pos = src.find("_connected = false;")
        self.assertNotEqual(disconnect_pos, -1)
        self.assertIn("_clearWarningLed()", src[disconnect_pos:disconnect_pos + 160])

        reset_pos = src.find("void resetToManualSafeState()")
        self.assertNotEqual(reset_pos, -1)
        self.assertIn("_clearWarningLed()", src[reset_pos:reset_pos + 260])

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
