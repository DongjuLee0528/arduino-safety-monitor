import threading
import unittest

from mpu.bridge_rpc import BridgeRPC, RPCError, RPCProtocolError, UnoQBridgeTransport


class FakeTransport:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = []

    def call(self, endpoint, *args, timeout=None):
        self.calls.append((endpoint, args, timeout))
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise TimeoutError("no response")
        return self.responses.pop(0)


def _bridge(responses=None, exc=None):
    return BridgeRPC(transport=FakeTransport(responses, exc))


class TestUnoQBridgeTransport(unittest.TestCase):
    def test_bridge_call_direct_return(self):
        class Bridge:
            @staticmethod
            def call(endpoint, *args, timeout=10):
                return {"type": "pong", "endpoint": endpoint, "args": args, "timeout": timeout}

        transport = UnoQBridgeTransport(bridge=Bridge)
        self.assertEqual(
            transport.call("asm_ping"),
            {"type": "pong", "endpoint": "asm_ping", "args": (), "timeout": 10},
        )

    def test_bridge_call_exception_becomes_connection_error(self):
        class Bridge:
            @staticmethod
            def call(endpoint, *args, timeout=10):
                raise RuntimeError("router unavailable")

        transport = UnoQBridgeTransport(bridge=Bridge)
        with self.assertRaises(ConnectionError):
            transport.call("asm_ping")

    def test_async_result_object_supported(self):
        class Result:
            def result(self):
                return {"type": "pong"}

        class Bridge:
            @staticmethod
            def call(endpoint, *args, timeout=10):
                return Result()

        transport = UnoQBridgeTransport(bridge=Bridge)
        self.assertEqual(transport.call("asm_ping"), {"type": "pong"})


class TestConnectionSemantics(unittest.TestCase):
    def test_connect_requires_valid_ping(self):
        bridge = _bridge([{"type": "pong"}])
        bridge.connect()
        self.assertTrue(bridge._connected)
        self.assertEqual(bridge.transport.calls[0], ("asm_ping", (), 1.0))
        bridge.disconnect()

    def test_connect_rejects_invalid_ping(self):
        bridge = _bridge([{"type": "motor_ack", "direction": "forward", "speed": 150}])
        with self.assertRaises(ConnectionError):
            bridge.connect()
        self.assertFalse(bridge._connected)

    def test_disconnect_marks_unhealthy(self):
        bridge = _bridge([{"type": "pong"}])
        bridge.connect()
        bridge.disconnect()
        self.assertFalse(bridge._connected)
        self.assertFalse(bridge._heartbeat_healthy)


class TestCommandMapping(unittest.TestCase):
    def test_ping_success(self):
        bridge = _bridge([{"type": "pong"}])
        self.assertTrue(bridge.ping())
        self.assertEqual(bridge.transport.calls, [("asm_ping", (), 1.0)])

    def test_motor_forward_mapping(self):
        bridge = _bridge([{"type": "motor_ack", "direction": "forward", "speed": 150}])
        self.assertTrue(bridge.motor_control("forward", 150))
        self.assertEqual(bridge.transport.calls, [("asm_motor", ("forward", 150), 1.0)])

    def test_motor_stop_mapping(self):
        bridge = _bridge([{"type": "motor_ack", "direction": "stop", "speed": 200}])
        self.assertTrue(bridge.motor_control("stop"))
        self.assertEqual(bridge.transport.calls, [("asm_motor", ("stop", 200), 1.0)])

    def test_mode_mapping(self):
        bridge = _bridge([{"type": "mode_ack", "mode": "manual"}])
        self.assertTrue(bridge.mode_control("manual"))
        self.assertEqual(bridge.transport.calls, [("asm_mode", ("manual",), 1.0)])

    def test_safe_reset_mapping(self):
        bridge = _bridge([{"type": "safe_reset_ack", "status": "ok", "mode": "manual"}])
        self.assertTrue(bridge.safe_reset())
        self.assertEqual(bridge.transport.calls, [("asm_safe_reset", (), 2.0)])

    def test_control_tick_mapping(self):
        bridge = _bridge([{"type": "control_tick_ack", "motion_authorized": True}])
        self.assertTrue(bridge.control_tick())
        self.assertEqual(bridge.transport.calls, [("asm_control_tick", (), 1.0)])

    def test_status_mapping(self):
        status = {
            "type": "status",
            "mode": "manual",
            "movement": "stopped",
            "motor_speed": 200,
            "motion_lease_active": False,
            "control_tick_fresh": False,
            "command_fresh": True,
            "warning_active": False,
            "safety_state": "safe",
            "distances": {"front": None, "rear": 20.0, "left": None, "right": 30.5},
        }
        bridge = _bridge([status])
        self.assertEqual(bridge.get_status(), status)
        self.assertEqual(bridge.transport.calls, [("asm_status", (), 1.0)])

    def test_led_mapping(self):
        bridge = _bridge([{"type": "led_ack", "color": "red"}])
        self.assertTrue(bridge.led_control("red"))
        self.assertEqual(bridge.transport.calls, [("asm_led", ("red",), 1.0)])

    def test_buzzer_unsupported_truthful(self):
        bridge = _bridge([{"type": "error", "error": "buzzer_not_supported"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.buzzer_control("off")
        self.assertEqual(ctx.exception.error_code, "buzzer_not_supported")
        self.assertEqual(bridge.transport.calls, [("asm_buzzer", ("off",), 1.0)])


class TestValidation(unittest.TestCase):
    def test_invalid_motor_direction_rejected_before_transport(self):
        bridge = _bridge()
        with self.assertRaises(ValueError):
            bridge.motor_control("diagonal", 150)
        self.assertEqual(bridge.transport.calls, [])

    def test_invalid_speed_rejected_before_transport(self):
        for bad in (True, False, 12.5, "150", None, -1, 256):
            with self.subTest(bad=bad):
                bridge = _bridge()
                with self.assertRaises(ValueError):
                    bridge.motor_control("forward", bad)
                self.assertEqual(bridge.transport.calls, [])

    def test_speed_bounds_allowed(self):
        for speed in (0, 255):
            with self.subTest(speed=speed):
                bridge = _bridge([{"type": "motor_ack", "direction": "forward", "speed": speed}])
                self.assertTrue(bridge.motor_control("forward", speed))

    def test_invalid_mode_rejected(self):
        bridge = _bridge()
        with self.assertRaises(ValueError):
            bridge.mode_control("park")
        self.assertEqual(bridge.transport.calls, [])

    def test_invalid_led_rejected(self):
        bridge = _bridge()
        with self.assertRaises(ValueError):
            bridge.led_control("blue")
        self.assertEqual(bridge.transport.calls, [])


class TestAckAndErrorSemantics(unittest.TestCase):
    def test_rpc_error_preserved(self):
        bridge = _bridge([{"type": "error", "error": "CMD_NOT_ALLOWED_IN_AUTO"}])
        with self.assertRaises(RPCError) as ctx:
            bridge.motor_control("forward", 150)
        self.assertEqual(ctx.exception.error_code, "CMD_NOT_ALLOWED_IN_AUTO")

    def test_wrong_ack_rejected(self):
        bridge = _bridge([{"type": "mode_ack", "mode": "manual"}])
        with self.assertRaises(RPCProtocolError):
            bridge.motor_control("forward", 150)

    def test_json_string_response_accepted(self):
        bridge = _bridge(['{"type":"pong"}'])
        self.assertTrue(bridge.ping())

    def test_non_json_string_rejected(self):
        bridge = _bridge(["not-json"])
        with self.assertRaises(RPCProtocolError):
            bridge.ping()

    def test_transport_exception_surfaces_as_controlled_failure(self):
        bridge = _bridge(exc=ConnectionError("router unavailable"))
        with self.assertRaises(ConnectionError):
            bridge.ping()

    def test_control_tick_missing_bool_rejected(self):
        bridge = _bridge([{"type": "control_tick_ack"}])
        with self.assertRaises(RPCProtocolError):
            bridge.control_tick()

    def test_invalid_status_rejected(self):
        bridge = _bridge([{"type": "status", "mode": "manual"}])
        with self.assertRaises(RPCProtocolError):
            bridge.get_status()

    def test_safe_reset_wrong_mode_rejected(self):
        bridge = _bridge([{"type": "safe_reset_ack", "status": "ok", "mode": "auto"}])
        with self.assertRaises(RPCProtocolError):
            bridge.safe_reset()


class TestSerializationWithLock(unittest.TestCase):
    def test_concurrent_requests_do_not_interleave(self):
        transport = FakeTransport([
            {"type": "pong"},
            {"type": "led_ack", "color": "red"},
        ])
        bridge = BridgeRPC(transport=transport)
        results = []
        errors = []

        def do_ping():
            try:
                results.append(bridge.ping())
            except Exception as exc:
                errors.append(exc)

        def do_led():
            try:
                results.append(bridge.led_control("red"))
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=do_ping)
        t2 = threading.Thread(target=do_led)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(transport.calls), 2)


class TestNoRawSerialRuntimePath(unittest.TestCase):
    def test_bridge_rpc_source_does_not_import_pyserial(self):
        import inspect
        import mpu.bridge_rpc as bridge_rpc

        src = inspect.getsource(bridge_rpc)
        self.assertNotIn("import serial", src)
        self.assertNotIn("serial.Serial", src)
        self.assertNotIn("/dev/ttyUSB", src)
        self.assertNotIn("/dev/ttyGS", src)
        self.assertNotIn("/dev/ttyHS", src)


if __name__ == "__main__":
    unittest.main()
