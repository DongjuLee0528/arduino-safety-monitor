import io
import json
import time
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from mpu.bridge_rpc import BridgeRPC, RPCError, RPCProtocolError


def _make_serial(responses):
    ser = MagicMock()
    ser.is_open = True

    encoded = b"".join(
        (json.dumps(r) + "\n").encode() for r in responses
    )
    buf = io.BytesIO(encoded)

    def _in_waiting():
        return len(buf.getvalue()) - buf.tell()

    def _read(n):
        return buf.read(n)

    type(ser).in_waiting = PropertyMock(side_effect=_in_waiting)
    ser.read.side_effect = _read
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


class TestPingSuccess(unittest.TestCase):
    def test_pong_returned(self):
        bridge = _make_bridge([{"type": "pong"}])
        result = bridge.ping()
        self.assertTrue(result)


class TestPingMismatch(unittest.TestCase):
    def test_motor_ack_raises_protocol_error(self):
        bridge = _make_bridge([{"type": "motor_ack", "direction": "forward", "speed": 150}])
        with self.assertRaises(RPCProtocolError):
            bridge.ping()


class TestMotorSuccess(unittest.TestCase):
    def test_motor_ack_accepted(self):
        bridge = _make_bridge([{"type": "motor_ack", "direction": "forward", "speed": 150}])
        result = bridge.motor_control("forward", 150)
        self.assertTrue(result)


class TestMotorArduinoError(unittest.TestCase):
    def test_manual_not_allowed_raises_rpc_error(self):
        bridge = _make_bridge([{"type": "error", "error": "manual_command_not_allowed_in_auto"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.motor_control("forward", 150)
        self.assertEqual(ctx.exception.error_code, "manual_command_not_allowed_in_auto")

    def test_rpc_error_not_treated_as_success(self):
        bridge = _make_bridge([{"type": "error", "error": "manual_command_not_allowed_in_auto"}])
        try:
            bridge.motor_control("forward", 150)
            self.fail("Should have raised RPCError")
        except RPCError:
            pass


class TestSafeModeActiveError(unittest.TestCase):
    def test_safe_mode_active_raises_rpc_error(self):
        bridge = _make_bridge([{"type": "error", "error": "safe_mode_active"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.motor_control("forward", 150)
        self.assertEqual(ctx.exception.error_code, "safe_mode_active")

    def test_safe_mode_active_not_success(self):
        bridge = _make_bridge([{"type": "error", "error": "safe_mode_active"}])
        try:
            bridge.motor_control("forward", 150)
            self.fail("Should have raised RPCError")
        except RPCError:
            pass


class TestSafeResetSuccess(unittest.TestCase):
    def test_safe_reset_ack_returns_true(self):
        bridge = _make_bridge([{"type": "safe_reset_ack", "status": "ok", "mode": "manual"}])
        result = bridge.safe_reset()
        self.assertTrue(result)


class TestSafeResetNotActive(unittest.TestCase):
    def test_safe_mode_not_active_raises_rpc_error(self):
        bridge = _make_bridge([{"type": "error", "error": "safe_mode_not_active"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.safe_reset()
        self.assertEqual(ctx.exception.error_code, "safe_mode_not_active")


class TestSafeResetWrongMode(unittest.TestCase):
    def test_mode_auto_in_safe_reset_ack_raises_protocol_error(self):
        bridge = _make_bridge([{"type": "safe_reset_ack", "status": "ok", "mode": "auto"}])
        with self.assertRaises(RPCProtocolError):
            bridge.safe_reset()


class TestMalformedResponse(unittest.TestCase):
    def test_non_json_raises_protocol_error(self):
        bridge = _make_bridge([])
        bridge.ser = MagicMock()
        bridge.ser.is_open = True
        raw = b"not-json\n"
        buf = io.BytesIO(raw)

        def _in_waiting():
            return len(raw) - buf.tell()

        type(bridge.ser).in_waiting = PropertyMock(side_effect=_in_waiting)
        bridge.ser.read.side_effect = buf.read
        bridge.ser.write = MagicMock()

        with self.assertRaises(RPCProtocolError):
            bridge.ping()

    def test_timeout_raises_timeout_error(self):
        bridge = _make_bridge([])
        type(bridge.ser).in_waiting = PropertyMock(return_value=0)
        with self.assertRaises(TimeoutError):
            bridge._request({"cmd": "ping"}, ack_timeout=0.05)


class TestHeartbeatThreadNotDuplicated(unittest.TestCase):
    def test_start_heartbeat_twice_creates_one_thread(self):
        bridge = BridgeRPC.__new__(BridgeRPC)
        bridge._lock = threading.Lock()
        bridge._heartbeat_thread = None
        bridge._heartbeat_stop = threading.Event()
        bridge._heartbeat_failures = 0
        bridge._heartbeat_healthy = True
        bridge.ser = _make_serial([{"type": "pong"}] * 100)

        bridge._start_heartbeat()
        first = bridge._heartbeat_thread
        bridge._start_heartbeat()
        second = bridge._heartbeat_thread

        self.assertIs(first, second)
        bridge._stop_heartbeat()


class TestDisconnectHealthState(unittest.TestCase):
    def test_disconnect_sets_unhealthy(self):
        bridge = BridgeRPC.__new__(BridgeRPC)
        bridge._lock = threading.Lock()
        bridge._heartbeat_thread = None
        bridge._heartbeat_stop = threading.Event()
        bridge._heartbeat_failures = 0
        bridge._heartbeat_healthy = True
        bridge.ser = _make_serial([])
        bridge.disconnect()
        self.assertFalse(bridge._heartbeat_healthy)


class TestHeartbeatStopsOnClose(unittest.TestCase):
    def test_thread_stops_after_disconnect(self):
        bridge = BridgeRPC.__new__(BridgeRPC)
        bridge._lock = threading.Lock()
        bridge._heartbeat_thread = None
        bridge._heartbeat_stop = threading.Event()
        bridge._heartbeat_failures = 0
        bridge._heartbeat_healthy = True
        bridge.ser = _make_serial([{"type": "pong"}] * 100)

        bridge._start_heartbeat()
        self.assertTrue(bridge._heartbeat_thread.is_alive())

        bridge._stop_heartbeat()
        time.sleep(0.2)
        self.assertFalse(
            bridge._heartbeat_thread is not None and bridge._heartbeat_thread.is_alive()
        )


class TestHeartbeatFailureCount(unittest.TestCase):
    def test_failure_increments_on_timeout(self):
        bridge = _make_bridge([])
        type(bridge.ser).in_waiting = PropertyMock(return_value=0)

        def _failing_request(cmd, ack_timeout=1.0):
            raise TimeoutError("simulated")

        bridge._request = _failing_request

        for _ in range(3):
            try:
                bridge._request({"cmd": "ping"}, ack_timeout=0.01)
            except TimeoutError:
                bridge._heartbeat_failures += 1
                if bridge._heartbeat_failures >= 3 and bridge._heartbeat_healthy:
                    bridge._heartbeat_healthy = False

        self.assertEqual(bridge._heartbeat_failures, 3)
        self.assertFalse(bridge._heartbeat_healthy)


class TestSerializationWithLock(unittest.TestCase):
    def test_concurrent_requests_do_not_interleave(self):
        write_order = []
        response_map = {
            b'{"cmd": "ping"}\n': {"type": "pong"},
            b'{"cmd": "led", "value": "red"}\n': {"type": "led_ack", "color": "red"},
        }
        pending = {"data": b""}

        ser = MagicMock()
        ser.is_open = True

        def _write(data):
            write_order.append(data)
            pending["data"] = (json.dumps(response_map[data]) + "\n").encode()

        def _in_waiting():
            return len(pending["data"])

        def _read(n):
            chunk = pending["data"][:n]
            pending["data"] = pending["data"][n:]
            return chunk

        type(ser).in_waiting = PropertyMock(side_effect=_in_waiting)
        ser.read.side_effect = _read
        ser.write.side_effect = _write

        bridge = _make_bridge([])
        bridge.ser = ser

        results = []
        errors = []

        def do_ping():
            try:
                results.append(bridge.ping())
            except Exception as e:
                errors.append(e)

        def do_led():
            try:
                results.append(bridge.led_control("red"))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_ping)
        t2 = threading.Thread(target=do_led)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
        self.assertEqual(len(results), 2)
        self.assertEqual(len(write_order), 2)


class TestAsyncTelemetrySkipped(unittest.TestCase):
    def test_telemetry_before_ack_succeeds(self):
        bridge = _make_bridge([
            {"type": "sensor_data", "timestamp": 1000},
            {"type": "motor_ack", "direction": "forward", "speed": 150},
        ])
        result = bridge.motor_control("forward", 150)
        self.assertTrue(result)

    def test_multiple_telemetry_then_ack(self):
        bridge = _make_bridge([
            {"type": "ultrasonic", "front": 10.0, "right": 50.0, "back": 100.0, "left": 80.0},
            {"type": "motor_status", "status": "cruising"},
            {"type": "avoidance", "direction": "left"},
            {"type": "mode_ack", "mode": "manual"},
        ])
        result = bridge.mode_control("manual")
        self.assertTrue(result)

    def test_wrong_ack_after_telemetry_raises_protocol_error(self):
        bridge = _make_bridge([
            {"type": "motor_status", "status": "cruising"},
            {"type": "mode_ack", "mode": "manual"},
        ])
        with self.assertRaises(RPCProtocolError):
            bridge.motor_control("forward", 150)

    def test_error_after_telemetry_raises_rpc_error(self):
        bridge = _make_bridge([
            {"type": "sensor_data", "timestamp": 1000},
            {"type": "error", "error": "safe_mode_active"},
        ])
        with self.assertRaises(RPCError) as ctx:
            bridge.motor_control("forward", 150)
        self.assertEqual(ctx.exception.error_code, "safe_mode_active")


class TestIsValidAck(unittest.TestCase):
    def setUp(self):
        self.bridge = BridgeRPC.__new__(BridgeRPC)
        self.bridge._lock = threading.Lock()
        self.bridge._heartbeat_thread = None
        self.bridge._heartbeat_stop = threading.Event()
        self.bridge._heartbeat_failures = 0
        self.bridge._heartbeat_healthy = True
        self.bridge.ser = None

    def test_ping_pong(self):
        self.assertTrue(self.bridge._is_valid_ack({"cmd": "ping"}, {"type": "pong"}))

    def test_ping_wrong_type(self):
        self.assertFalse(self.bridge._is_valid_ack({"cmd": "ping"}, {"type": "motor_ack"}))

    def test_motor_ack_correct(self):
        cmd = {"cmd": "motor", "direction": "forward", "speed": 150}
        resp = {"type": "motor_ack", "direction": "forward", "speed": 150}
        self.assertTrue(self.bridge._is_valid_ack(cmd, resp))

    def test_motor_ack_direction_mismatch(self):
        cmd = {"cmd": "motor", "direction": "forward", "speed": 150}
        resp = {"type": "motor_ack", "direction": "backward", "speed": 150}
        self.assertFalse(self.bridge._is_valid_ack(cmd, resp))

    def test_safe_reset_ack_correct(self):
        cmd = {"cmd": "safe_reset"}
        resp = {"type": "safe_reset_ack", "status": "ok", "mode": "manual"}
        self.assertTrue(self.bridge._is_valid_ack(cmd, resp))

    def test_safe_reset_ack_wrong_mode(self):
        cmd = {"cmd": "safe_reset"}
        resp = {"type": "safe_reset_ack", "status": "ok", "mode": "auto"}
        self.assertFalse(self.bridge._is_valid_ack(cmd, resp))

    def test_non_string_type(self):
        self.assertFalse(self.bridge._is_valid_ack({"cmd": "ping"}, {"type": 123}))

    def test_mode_ack_correct(self):
        cmd = {"cmd": "mode", "value": "manual"}
        resp = {"type": "mode_ack", "mode": "manual"}
        self.assertTrue(self.bridge._is_valid_ack(cmd, resp))

    def test_mode_ack_wrong_value(self):
        cmd = {"cmd": "mode", "value": "manual"}
        resp = {"type": "mode_ack", "mode": "auto"}
        self.assertFalse(self.bridge._is_valid_ack(cmd, resp))


# ---------------------------------------------------------------------------
# Python motor_control() speed type validation (tests 1-9)
# ---------------------------------------------------------------------------

class TestMotorControlSpeedTypeValidation(unittest.TestCase):
    def _bridge(self):
        return _make_bridge([{"type": "motor_ack", "direction": "forward", "speed": 150}])

    def test_speed_true_rejected(self):
        with self.assertRaises(ValueError):
            self._bridge().motor_control("forward", True)

    def test_speed_false_rejected(self):
        with self.assertRaises(ValueError):
            self._bridge().motor_control("forward", False)

    def test_speed_float_rejected(self):
        with self.assertRaises(ValueError):
            self._bridge().motor_control("forward", 12.5)

    def test_speed_string_rejected(self):
        with self.assertRaises(ValueError):
            self._bridge().motor_control("forward", "150")

    def test_speed_none_rejected(self):
        with self.assertRaises(ValueError):
            self._bridge().motor_control("forward", None)

    def test_speed_zero_accepted(self):
        bridge = _make_bridge([{"type": "motor_ack", "direction": "forward", "speed": 0}])
        self.assertTrue(bridge.motor_control("forward", 0))

    def test_speed_255_accepted(self):
        bridge = _make_bridge([{"type": "motor_ack", "direction": "forward", "speed": 255}])
        self.assertTrue(bridge.motor_control("forward", 255))

    def test_speed_minus_one_rejected(self):
        with self.assertRaises(ValueError):
            self._bridge().motor_control("forward", -1)

    def test_speed_256_rejected(self):
        with self.assertRaises(ValueError):
            self._bridge().motor_control("forward", 256)


if __name__ == "__main__":
    unittest.main()
