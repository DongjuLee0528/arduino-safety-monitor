"""
Motion Lease / Control-Loop Watchdog regression tests.

Coverage:
  Part A – MCU CommunicationManager state-machine simulation
  Part B – Arduino source structural assertions (supplemental)
  Part C – BridgeRPC control_tick Python method
  Part D – main.py control_tick integration

Part A uses a faithful host-side Python simulation of CommunicationManager
behaviour so that exact command sequences can be tested without native MCU
execution.  The simulator is NOT the actual C++ code; it reproduces the
specified state-machine transitions verified by the structural assertions in
Part B.

Arduino tests are NOT native MCU execution.
"""

import io
import json
import math
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mpu.bridge_rpc import BridgeRPC, RPCError, RPCProtocolError
from mpu.dashboard_state import DashboardState, EventType, HelmetResult
from mpu.main import HelmetDetectionSystem, _EventSuppressor, _CONTROL_TICK_INTERVAL


# ---------------------------------------------------------------------------
# Part A – MCU CommunicationManager simulator
# ---------------------------------------------------------------------------

_MOTOR_SPEED_DEFAULT = 150
_MOTION_LEASE_TIMEOUT_MS = 1000


class _CommSim:
    """
    Host-side simulation of CommunicationManager state transitions.

    All millis() calls are replaced by a controllable clock (_now_ms).
    Call advance(ms) to move time forward without running real timers.
    Call process(json_str) to inject one command line.
    The returned list contains the JSON ACK strings that would be sent.
    """

    def __init__(self):
        self._now_ms = 1000

        self._mode = "manual"
        self._pending_move = "none"
        self._pending_speed = _MOTOR_SPEED_DEFAULT
        self._has_pending = False
        self._stop_latched = False

        self._motion_lease_active = False
        self._last_motion_lease_ms = 0

        self._connected = True
        self._last_rx_ms = self._now_ms

    # ------------------------------------------------------------------
    # Time control
    # ------------------------------------------------------------------

    def advance(self, ms: int):
        self._now_ms += ms

    # ------------------------------------------------------------------
    # Lease helpers
    # ------------------------------------------------------------------

    def _start_motion_lease(self):
        self._motion_lease_active = True
        self._last_motion_lease_ms = self._now_ms

    def _renew_motion_lease(self):
        self._last_motion_lease_ms = self._now_ms

    def _clear_motion_lease(self):
        self._motion_lease_active = False
        self._last_motion_lease_ms = 0

    def _lease_expired(self):
        if not self._motion_lease_active:
            return False
        return (self._now_ms - self._last_motion_lease_ms) > _MOTION_LEASE_TIMEOUT_MS

    def _expire_lease(self):
        self._clear_motion_lease()
        self._mode = "manual"
        self._pending_move = "stop"
        self._pending_speed = _MOTOR_SPEED_DEFAULT
        self._has_pending = True
        self._stop_latched = True

    # ------------------------------------------------------------------
    # update() – called once per loop before process_line
    # ------------------------------------------------------------------

    def update_tick(self):
        """Simulate the start-of-update() lease expiry check."""
        if self._lease_expired():
            self._expire_lease()

    # ------------------------------------------------------------------
    # Command processing
    # ------------------------------------------------------------------

    def process(self, cmd_json: str) -> list:
        """
        Process one newline-terminated command JSON string.
        Returns list of response strings that would be sent over serial.
        """
        self.update_tick()

        try:
            cmd = json.loads(cmd_json)
        except json.JSONDecodeError:
            return ['{"type":"error","error":"MISSING_CMD"}']

        cmd_type = cmd.get("cmd", "")
        self._last_rx_ms = self._now_ms

        if cmd_type == "ping":
            return ['{"type":"pong"}']

        elif cmd_type == "motor":
            direction = cmd.get("direction", "")
            speed = cmd.get("speed", _MOTOR_SPEED_DEFAULT)

            if direction not in ("forward", "backward", "left", "right", "stop"):
                return ['{"type":"error","error":"UNKNOWN_DIRECTION"}']

            if self._mode == "auto" and direction != "stop":
                return ['{"type":"error","error":"CMD_NOT_ALLOWED_IN_AUTO"}']

            if direction == "stop":
                self._mode = "manual"
                self._pending_move = "stop"
                self._pending_speed = _MOTOR_SPEED_DEFAULT
                self._has_pending = True
                self._stop_latched = True
                self._clear_motion_lease()
            elif not self._stop_latched:
                self._pending_move = direction
                self._pending_speed = speed
                self._has_pending = True
                self._start_motion_lease()

            return [f'{{"type":"motor_ack","direction":"{direction}","speed":{speed}}}']

        elif cmd_type == "mode":
            value = cmd.get("value", "")
            if value == "auto":
                if not self._stop_latched:
                    self._mode = "auto"
                    self._pending_move = "none"
                    self._pending_speed = _MOTOR_SPEED_DEFAULT
                    self._has_pending = False
                    self._start_motion_lease()
            elif value == "manual":
                if self._mode != "manual":
                    self._pending_move = "stop"
                    self._pending_speed = _MOTOR_SPEED_DEFAULT
                    self._has_pending = True
                self._mode = "manual"
                self._clear_motion_lease()
            else:
                return ['{"type":"error","error":"UNKNOWN_MODE"}']
            return [f'{{"type":"mode_ack","mode":"{value}"}}']

        elif cmd_type == "control_tick":
            if self._motion_lease_active:
                self._renew_motion_lease()
                return ['{"type":"control_tick_ack","motion_authorized":true}']
            else:
                return ['{"type":"control_tick_ack","motion_authorized":false}']

        elif cmd_type == "safe_reset":
            self._clear_motion_lease()
            self._mode = "manual"
            self._pending_move = "stop"
            self._pending_speed = _MOTOR_SPEED_DEFAULT
            self._has_pending = True
            self._stop_latched = True
            return ['{"type":"safe_reset_ack","status":"ok","mode":"manual"}']

        else:
            return ['{"type":"error","error":"UNKNOWN_CMD"}']

    def consume_pending_move(self):
        """Simulate RobotController consuming the pending move."""
        self._has_pending = False
        self._stop_latched = False
        cmd = self._pending_move
        self._pending_move = "none"
        self._pending_speed = _MOTOR_SPEED_DEFAULT
        return cmd

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def mode(self):
        return self._mode

    @property
    def pending_move(self):
        return self._pending_move

    @property
    def has_pending(self):
        return self._has_pending

    @property
    def stop_latched(self):
        return self._stop_latched

    @property
    def motion_lease_active(self):
        return self._motion_lease_active


# ---------------------------------------------------------------------------
# Part A tests – MCU state machine
# ---------------------------------------------------------------------------

class TestLeaseStartAndRenewal(unittest.TestCase):

    def test_forward_starts_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertTrue(sim.motion_lease_active)

    def test_backward_starts_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"backward","speed":150}')
        self.assertTrue(sim.motion_lease_active)

    def test_left_starts_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"left","speed":150}')
        self.assertTrue(sim.motion_lease_active)

    def test_right_starts_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"right","speed":150}')
        self.assertTrue(sim.motion_lease_active)

    def test_auto_mode_starts_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        self.assertTrue(sim.motion_lease_active)

    def test_control_tick_renews_active_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(800)
        resps = sim.process('{"cmd":"control_tick"}')
        self.assertTrue(sim.motion_lease_active)
        self.assertIn('"motion_authorized":true', resps[0])

    def test_repeated_movement_renews_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        sim.advance(800)
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        sim.advance(800)
        self.assertTrue(sim.motion_lease_active)

    def test_ping_does_not_start_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"ping"}')
        self.assertFalse(sim.motion_lease_active)

    def test_ping_does_not_renew_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(800)
        sim.process('{"cmd":"ping"}')
        sim.advance(300)
        sim.update_tick()
        self.assertFalse(sim.motion_lease_active)
        self.assertEqual(sim.pending_move, "stop")

    def test_motor_stop_clears_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertTrue(sim.motion_lease_active)
        sim.process('{"cmd":"motor","direction":"stop","speed":150}')
        self.assertFalse(sim.motion_lease_active)

    def test_safe_reset_clears_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        self.assertTrue(sim.motion_lease_active)
        sim.process('{"cmd":"safe_reset"}')
        self.assertFalse(sim.motion_lease_active)

    def test_control_tick_inactive_does_not_start_lease(self):
        sim = _CommSim()
        resps = sim.process('{"cmd":"control_tick"}')
        self.assertFalse(sim.motion_lease_active)
        self.assertIn('"motion_authorized":false', resps[0])

    def test_control_tick_inactive_no_movement(self):
        sim = _CommSim()
        sim.process('{"cmd":"control_tick"}')
        self.assertFalse(sim.has_pending)
        self.assertEqual(sim.mode, "manual")

    def test_mode_manual_clears_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        self.assertTrue(sim.motion_lease_active)
        sim.process('{"cmd":"mode","value":"manual"}')
        self.assertFalse(sim.motion_lease_active)

    def test_mode_manual_control_tick_returns_false(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.process('{"cmd":"mode","value":"manual"}')
        resps = sim.process('{"cmd":"control_tick"}')
        self.assertIn('"motion_authorized":false', resps[0])

    def test_mode_manual_requires_fresh_movement_after(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.process('{"cmd":"mode","value":"manual"}')
        sim.consume_pending_move()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertEqual(sim.pending_move, "forward")
        self.assertTrue(sim.motion_lease_active)

    def test_mode_manual_clears_lease_even_if_already_manual(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertTrue(sim.motion_lease_active)
        sim.process('{"cmd":"mode","value":"manual"}')
        self.assertFalse(sim.motion_lease_active)


class TestLeaseExpiry(unittest.TestCase):

    def test_lease_expires_and_latches_stop(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        self.assertFalse(sim.motion_lease_active)
        self.assertEqual(sim.pending_move, "stop")
        self.assertTrue(sim.stop_latched)
        self.assertEqual(sim.mode, "manual")

    def test_expiry_forces_manual(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        self.assertEqual(sim.mode, "manual")

    def test_expiry_resets_pending_speed(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"forward","speed":200}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        self.assertEqual(sim.pending_move, "stop")

    def test_expiry_before_buffered_auto_blocks_auto(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        resps = sim.process('{"cmd":"mode","value":"auto"}')
        self.assertEqual(sim.mode, "manual")
        self.assertTrue(sim.stop_latched)

    def test_expiry_before_buffered_forward_blocks_forward(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertEqual(sim.pending_move, "stop")
        self.assertTrue(sim.stop_latched)

    def test_stop_after_expiry_remains_stop(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        sim.process('{"cmd":"motor","direction":"stop","speed":150}')
        self.assertEqual(sim.pending_move, "stop")

    def test_control_tick_after_expiry_does_not_restore(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        sim.consume_pending_move()
        resps = sim.process('{"cmd":"control_tick"}')
        self.assertFalse(sim.motion_lease_active)
        self.assertIn('"motion_authorized":false', resps[0])

    def test_new_explicit_movement_after_expiry_works(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        sim.consume_pending_move()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertEqual(sim.pending_move, "forward")
        self.assertTrue(sim.motion_lease_active)

    def test_new_explicit_auto_after_expiry_works(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        sim.consume_pending_move()
        sim.process('{"cmd":"mode","value":"auto"}')
        self.assertEqual(sim.mode, "auto")
        self.assertTrue(sim.motion_lease_active)


class TestStopAndSafeResetInteractionWithLease(unittest.TestCase):

    def test_stop_works_regardless_of_lease_state(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"stop","speed":150}')
        self.assertEqual(sim.pending_move, "stop")
        self.assertTrue(sim.stop_latched)

    def test_stop_executes_exactly_once(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        sim.process('{"cmd":"motor","direction":"stop","speed":150}')
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertEqual(sim.pending_move, "stop")
        consumed = sim.consume_pending_move()
        self.assertEqual(consumed, "stop")
        self.assertFalse(sim.stop_latched)

    def test_stop_auto_batch_yields_stop(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.process('{"cmd":"motor","direction":"stop","speed":150}')
        sim.process('{"cmd":"mode","value":"auto"}')
        self.assertEqual(sim.mode, "manual")
        self.assertEqual(sim.pending_move, "stop")
        self.assertTrue(sim.stop_latched)

    def test_stop_forward_batch_yields_stop(self):
        sim = _CommSim()
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        sim.process('{"cmd":"motor","direction":"stop","speed":150}')
        sim.process('{"cmd":"motor","direction":"forward","speed":150}')
        self.assertEqual(sim.pending_move, "stop")

    def test_auto_never_runs_before_stop_consumed(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.process('{"cmd":"motor","direction":"stop","speed":150}')
        sim.process('{"cmd":"mode","value":"auto"}')
        self.assertEqual(sim.mode, "manual")

    def test_safe_reset_latches_stop(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.process('{"cmd":"safe_reset"}')
        self.assertEqual(sim.pending_move, "stop")
        self.assertTrue(sim.stop_latched)
        self.assertEqual(sim.mode, "manual")


class TestReconnectAndPingPolicy(unittest.TestCase):

    def test_reconnect_ping_does_not_restore_lease(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        sim.consume_pending_move()
        sim.process('{"cmd":"ping"}')
        self.assertFalse(sim.motion_lease_active)

    def test_reconnect_control_tick_alone_does_not_restore(self):
        sim = _CommSim()
        sim.process('{"cmd":"mode","value":"auto"}')
        sim.advance(_MOTION_LEASE_TIMEOUT_MS + 1)
        sim.update_tick()
        sim.consume_pending_move()
        resps = sim.process('{"cmd":"control_tick"}')
        self.assertFalse(sim.motion_lease_active)
        self.assertIn('"motion_authorized":false', resps[0])


# ---------------------------------------------------------------------------
# Part B – Arduino source structural assertions (supplemental)
# ---------------------------------------------------------------------------

def _comm_h_src():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "arduino", "comm.h")
    )
    with open(path) as f:
        return f.read()


def _config_h_src():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "arduino", "config.h")
    )
    with open(path) as f:
        return f.read()


class TestMotionLeaseStructural(unittest.TestCase):

    def setUp(self):
        self.src = _comm_h_src()
        self.cfg = _config_h_src()

    def test_s01_motion_lease_timeout_in_config(self):
        self.assertIn("MOTION_LEASE_TIMEOUT_MS", self.cfg)

    def test_s02_motion_lease_active_field(self):
        self.assertIn("_motionLeaseActive", self.src)

    def test_s03_last_motion_lease_time_field(self):
        self.assertIn("_lastMotionLeaseTime", self.src)

    def test_s04_start_motion_lease_helper(self):
        self.assertIn("_startMotionLease()", self.src)

    def test_s05_renew_motion_lease_helper(self):
        self.assertIn("_renewMotionLease()", self.src)

    def test_s06_clear_motion_lease_helper(self):
        self.assertIn("_clearMotionLease()", self.src)

    def test_s07_motion_lease_expired_helper(self):
        self.assertIn("_motionLeaseExpired()", self.src)

    def test_s08_expire_lease_called_in_update(self):
        update_pos = self.src.find("void update()")
        self.assertNotEqual(update_pos, -1)
        update_snippet = self.src[update_pos:update_pos + 500]
        self.assertIn("_motionLeaseExpired()", update_snippet,
                      "Lease expiry must be checked inside update()")
        self.assertIn("_expireLease()", update_snippet)

    def test_s09_expiry_checked_before_serial_drain(self):
        update_pos = self.src.find("void update()")
        update_snippet = self.src[update_pos:update_pos + 600]
        expire_pos = update_snippet.find("_motionLeaseExpired()")
        serial_pos = update_snippet.find("Serial.available()")
        self.assertLess(expire_pos, serial_pos,
                        "Lease expiry check must precede Serial.available() loop")

    def test_s10_control_tick_command_handled(self):
        self.assertIn('"control_tick"', self.src)
        self.assertIn("control_tick_ack", self.src)

    def test_s11_control_tick_renews_only_when_active(self):
        tick_pos = self.src.find('strcmp(cmd, "control_tick")')
        self.assertNotEqual(tick_pos, -1, 'control_tick command dispatch must exist in code')
        tick_snippet = self.src[tick_pos:tick_pos + 400]
        self.assertIn("_renewMotionLease()", tick_snippet)
        self.assertIn("_motionLeaseActive", tick_snippet)

    def test_s12_ping_has_no_lease_call(self):
        ping_pos = self.src.find('"ping"')
        ping_snippet = self.src[ping_pos:ping_pos + 100]
        self.assertNotIn("_startMotionLease", ping_snippet)
        self.assertNotIn("_renewMotionLease", ping_snippet)

    def test_s13_stop_clears_lease(self):
        stop_pos = self.src.find("if (mc == MOVE_STOP)")
        self.assertNotEqual(stop_pos, -1)
        stop_snippet = self.src[stop_pos:stop_pos + 300]
        self.assertIn("_clearMotionLease()", stop_snippet)

    def test_s14_non_stop_move_starts_lease(self):
        motor_block_start = self.src.find('} else if (strcmp(cmd, "motor") == 0) {')
        motor_block_end = self.src.find("\n        } else if (strcmp(cmd", motor_block_start + 1)
        if motor_block_end == -1:
            motor_block_end = self.src.find("\n        } else {", motor_block_start + 1)
        block = self.src[motor_block_start:motor_block_end]
        self.assertIn("_startMotionLease()", block,
                      "Non-STOP motor move must call _startMotionLease()")

    def test_s15_auto_mode_starts_lease(self):
        mode_block_start = self.src.find('} else if (strcmp(cmd, "mode") == 0) {')
        mode_block_end = self.src.find("\n        } else if (strcmp(cmd", mode_block_start + 1)
        if mode_block_end == -1:
            mode_block_end = self.src.find("\n        } else {", mode_block_start + 1)
        block = self.src[mode_block_start:mode_block_end]
        self.assertIn("_startMotionLease()", block,
                      "mode=auto must call _startMotionLease()")

    def test_s16_safe_reset_clears_lease(self):
        sr_pos = self.src.find('} else if (strcmp(cmd, "safe_reset") == 0) {')
        self.assertNotEqual(sr_pos, -1)
        sr_snippet = self.src[sr_pos:sr_pos + 300]
        self.assertIn("_clearMotionLease()", sr_snippet)

    def test_s17_disconnect_clears_lease(self):
        update_pos = self.src.find("void update()")
        update_snippet = self.src[update_pos:update_pos + 300]
        self.assertIn("_clearMotionLease()", update_snippet,
                      "Serial timeout / disconnect must clear motion lease in update()")

    def test_s18_reset_to_manual_clears_lease(self):
        reset_pos = self.src.find("void resetToManualSafeState()")
        self.assertNotEqual(reset_pos, -1)
        reset_snippet = self.src[reset_pos:reset_pos + 300]
        self.assertIn("_clearMotionLease()", reset_snippet)

    def test_s19_is_motion_authorized_accessor_exists(self):
        self.assertIn("isMotionAuthorized()", self.src)

    def test_s20_motion_lease_active_initialised_false(self):
        ctor_pos = self.src.find("CommunicationManager()")
        ctor_snippet = self.src[ctor_pos:ctor_pos + 500]
        self.assertIn("_motionLeaseActive(false)", ctor_snippet)

    def test_s21_expire_lease_sets_stop_latched(self):
        expire_pos = self.src.find("void _expireLease()")
        self.assertNotEqual(expire_pos, -1)
        expire_snippet = self.src[expire_pos:expire_pos + 300]
        self.assertIn("_stopLatched  = true", expire_snippet)
        self.assertIn("MOVE_STOP", expire_snippet)
        self.assertIn("MODE_MANUAL", expire_snippet)

    def test_s22_mode_manual_clears_lease(self):
        mode_start = self.src.find('} else if (strcmp(cmd, "mode") == 0) {')
        mode_end   = self.src.find("\n        } else if (strcmp(cmd", mode_start + 1)
        if mode_end == -1:
            mode_end = self.src.find("\n        } else {", mode_start + 1)
        block = self.src[mode_start:mode_end]
        manual_pos = block.find('strcmp(value, "manual")')
        self.assertNotEqual(manual_pos, -1)
        manual_branch = block[manual_pos:manual_pos + 400]
        self.assertIn("_clearMotionLease()", manual_branch,
                      "mode=manual must call _clearMotionLease()")


# ---------------------------------------------------------------------------
# Part C – BridgeRPC control_tick Python method
# ---------------------------------------------------------------------------

def _make_serial(responses):
    ser = MagicMock()
    ser.is_open = True
    encoded = b"".join((json.dumps(r) + "\n").encode() for r in responses)
    buf = io.BytesIO(encoded)

    def _in_waiting():
        return len(buf.getvalue()) - buf.tell()

    type(ser).in_waiting = property(lambda s: len(buf.getvalue()) - buf.tell())
    from unittest.mock import PropertyMock
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


class TestBridgeControlTick(unittest.TestCase):

    def test_ct_authorized_true_returns_true(self):
        bridge = _make_bridge([{"type": "control_tick_ack", "motion_authorized": True}])
        result = bridge.control_tick()
        self.assertTrue(result)

    def test_ct_authorized_false_returns_false(self):
        bridge = _make_bridge([{"type": "control_tick_ack", "motion_authorized": False}])
        result = bridge.control_tick()
        self.assertFalse(result)

    def test_ct_request_shape(self):
        bridge = _make_bridge([{"type": "control_tick_ack", "motion_authorized": True}])
        bridge.control_tick()
        written = bridge.ser.write.call_args[0][0].decode()
        payload = json.loads(written.strip())
        self.assertEqual(payload, {"cmd": "control_tick"})

    def test_ct_arduino_error_preserved(self):
        bridge = _make_bridge([{"type": "error", "error": "UNKNOWN_CMD"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.control_tick()
        self.assertEqual(ctx.exception.error_code, "UNKNOWN_CMD")

    def test_ct_malformed_ack_rejected(self):
        bridge = _make_bridge([{"type": "control_tick_ack"}])
        with self.assertRaises(RPCProtocolError):
            bridge.control_tick()

    def test_ct_wrong_type_rejected(self):
        bridge = _make_bridge([{"type": "pong"}])
        with self.assertRaises(RPCProtocolError):
            bridge.control_tick()

    def test_ct_uses_request_lock(self):
        bridge = _make_bridge([{"type": "control_tick_ack", "motion_authorized": True}])
        acquired = []
        original_request = bridge._request

        def tracking_request(cmd, ack_timeout=1.0):
            acquired.append(bridge._lock.locked())
            return original_request(cmd, ack_timeout)

        bridge._request = tracking_request
        bridge.control_tick()

    def test_ct_timeout_raises(self):
        bridge = _make_bridge([])
        from unittest.mock import PropertyMock
        type(bridge.ser).in_waiting = PropertyMock(return_value=0)
        with self.assertRaises(TimeoutError):
            bridge._request({"cmd": "control_tick"}, ack_timeout=0.05)

    def test_heartbeat_loop_sends_ping_not_control_tick(self):
        import inspect
        src = inspect.getsource(BridgeRPC._heartbeat_loop)
        self.assertIn('"ping"', src)
        self.assertNotIn("control_tick", src)

    def test_valid_ack_true(self):
        bridge = _make_bridge([])
        cmd = {"cmd": "control_tick"}
        resp = {"type": "control_tick_ack", "motion_authorized": True}
        self.assertTrue(bridge._is_valid_ack(cmd, resp))

    def test_valid_ack_false(self):
        bridge = _make_bridge([])
        cmd = {"cmd": "control_tick"}
        resp = {"type": "control_tick_ack", "motion_authorized": False}
        self.assertTrue(bridge._is_valid_ack(cmd, resp))

    def test_ack_missing_motion_authorized_rejected(self):
        bridge = _make_bridge([])
        cmd = {"cmd": "control_tick"}
        resp = {"type": "control_tick_ack"}
        self.assertFalse(bridge._is_valid_ack(cmd, resp))

    def test_ack_motion_authorized_string_rejected(self):
        bridge = _make_bridge([])
        cmd = {"cmd": "control_tick"}
        resp = {"type": "control_tick_ack", "motion_authorized": "true"}
        self.assertFalse(bridge._is_valid_ack(cmd, resp))


# ---------------------------------------------------------------------------
# Part D – main.py control_tick integration
# ---------------------------------------------------------------------------

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
    system._last_tick_time = 0.0
    return system


class TestMainControlTick(unittest.TestCase):

    def _run_loop_iterations(self, system, n_frames, tick_interval=0.0):
        """Drive the start() loop body manually for n_frames."""
        system.running = True
        frame = np.zeros((480, 640, 3), dtype="uint8")
        system.camera.capture_frame.return_value = frame
        system.person_detector.detect.return_value = []

        import mpu.main as main_mod
        orig_interval = main_mod._CONTROL_TICK_INTERVAL
        main_mod._CONTROL_TICK_INTERVAL = tick_interval
        try:
            for _ in range(n_frames):
                now = time.monotonic()
                if now - system._last_tick_time >= main_mod._CONTROL_TICK_INTERVAL:
                    try:
                        system.bridge_rpc.control_tick()
                    except Exception:
                        pass
                    system._last_tick_time = time.monotonic()
                system.process_frame(frame)
        finally:
            main_mod._CONTROL_TICK_INTERVAL = orig_interval

    def test_tick_sent_on_first_frame(self):
        system = _make_system()
        system.bridge_rpc.control_tick.return_value = True
        self._run_loop_iterations(system, 1, tick_interval=0.0)
        system.bridge_rpc.control_tick.assert_called()

    def test_tick_not_sent_every_frame_when_interval_large(self):
        system = _make_system()
        system.bridge_rpc.control_tick.return_value = True
        system._last_tick_time = time.monotonic()
        self._run_loop_iterations(system, 5, tick_interval=999.0)
        system.bridge_rpc.control_tick.assert_not_called()

    def test_tick_failure_does_not_crash_loop(self):
        system = _make_system()
        system.bridge_rpc.control_tick.side_effect = RuntimeError("serial fail")
        try:
            self._run_loop_iterations(system, 2, tick_interval=0.0)
        except RuntimeError:
            self.fail("control_tick exception must not propagate from the loop")

    def test_tick_not_emitted_from_heartbeat(self):
        import inspect
        from mpu.bridge_rpc import BridgeRPC
        src = inspect.getsource(BridgeRPC._heartbeat_loop)
        self.assertNotIn("control_tick", src)

    def test_existing_detection_behavior_unchanged(self):
        system = _make_system()
        system.bridge_rpc.control_tick.return_value = True
        frame = np.zeros((480, 640, 3), dtype="uint8")
        system.person_detector.detect.return_value = [
            {"bbox": [10, 20, 80, 160], "confidence": 0.9}
        ]
        system.helmet_classifier.predict.return_value = {
            "label": "helmet", "confidence": 0.95
        }
        system.process_frame(frame)
        snap = system.dashboard.snapshot()["detection"]
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], "helmet")

    def test_last_tick_time_attribute_exists(self):
        with patch("mpu.main.CameraCapture"), \
             patch("mpu.main.PersonDetector"), \
             patch("mpu.main.HelmetClassifier"), \
             patch("mpu.main.BridgeRPC"), \
             patch("mpu.main.Sender"), \
             patch("mpu.main.AlertManager"):
            system = HelmetDetectionSystem(port="/dev/null", server_url="http://localhost")
        self.assertTrue(hasattr(system, "_last_tick_time"))
        self.assertEqual(system._last_tick_time, 0.0)

    def test_control_tick_interval_constant_exists(self):
        import mpu.main as main_mod
        self.assertTrue(hasattr(main_mod, "_CONTROL_TICK_INTERVAL"))
        self.assertAlmostEqual(main_mod._CONTROL_TICK_INTERVAL, 0.25)


if __name__ == "__main__":
    unittest.main()
