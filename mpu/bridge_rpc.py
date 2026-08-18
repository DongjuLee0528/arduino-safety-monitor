"""
Bridge RPC Communication Module

Provides the Python-side MCU command boundary for the Arduino UNO Q App Lab
runtime.  The public BridgeRPC methods preserve the application command
contract while the transport uses the official UNO Q App Lab Bridge IPC path
instead of raw serial device nodes.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 0.5
_HEARTBEAT_MAX_FAILURES = 3

_ENDPOINTS = {
    "ping": "asm_ping",
    "motor": "asm_motor",
    "mode": "asm_mode",
    "safe_reset": "asm_safe_reset",
    "control_tick": "asm_control_tick",
    "status": "asm_status",
    "ultrasonic_diag": "asm_ultrasonic_diag",
    "led": "asm_led",
    "buzzer": "asm_buzzer",
}


class RPCError(Exception):
    def __init__(self, command: Dict[str, Any], error_code: str, response: Dict[str, Any]):
        self.command = command
        self.error_code = error_code
        self.response = response
        super().__init__(
            f"Arduino returned error '{error_code}' for command '{command.get('cmd')}'"
        )


class RPCProtocolError(Exception):
    def __init__(self, command: Dict[str, Any], response: Any, reason: str):
        self.command = command
        self.response = response
        self.reason = reason
        super().__init__(
            f"Protocol error for command '{command.get('cmd') if isinstance(command, dict) else command}': {reason}"
        )


class UnoQBridgeTransport:
    """
    Thin adapter around the App Lab Bridge object.

    App Lab provides `Bridge.call(name, *args)` from arduino.app_utils.  The
    dependency is imported lazily so local unit tests and hardware-free dev mode
    do not require the App Lab runtime package to be installed.
    """

    def __init__(self, bridge: Any = None):
        self._bridge = bridge

    @property
    def bridge(self) -> Any:
        if self._bridge is None:
            try:
                from arduino.app_utils import Bridge  # type: ignore
            except Exception as exc:
                raise ConnectionError(
                    "Arduino App Lab Bridge is unavailable; run inside UNO Q App Lab "
                    "or inject a test transport"
                ) from exc
            self._bridge = Bridge
        return self._bridge

    def call(self, endpoint: str, *args: Any, timeout: float = 10) -> Any:
        try:
            result = self.bridge.call(endpoint, *args, timeout=timeout)
        except Exception as exc:
            raise ConnectionError(f"Bridge.call('{endpoint}') failed: {exc}") from exc

        # Some Bridge implementations expose an async call object.  Support that
        # shape without requiring it so tests and the documented direct-return API
        # both work.
        result_attr = getattr(result, "result", None)
        if callable(result_attr):
            try:
                return result_attr()
            except TypeError:
                # If the runtime exposes result as a method requiring arguments,
                # surface the transport failure clearly instead of fabricating ACKs.
                raise ConnectionError(
                    f"Bridge.call('{endpoint}') returned an unsupported async result object"
                )
            except Exception as exc:
                raise ConnectionError(f"Bridge.call('{endpoint}') result failed: {exc}") from exc
        return result


class BridgeRPC:
    """
    High-level MCU command client for robot control.

    Public methods intentionally remain stable for application code:
    ping, led_control, buzzer_control, motor_control, mode_control,
    safe_reset, control_tick, get_status, connect, and disconnect.
    """

    def __init__(self, port: Optional[str] = None, transport: Optional[UnoQBridgeTransport] = None):
        # `port` is retained for caller compatibility with the former raw serial
        # implementation.  UNO Q App Lab production transport ignores device paths.
        if transport is None and port is not None and hasattr(port, "call"):
            transport = port  # Backward-compatible test injection.
            port = None
        self.port = port or "unoq-bridge"
        self.transport = transport or UnoQBridgeTransport()
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_failures = 0
        self._heartbeat_healthy = True
        self._connected = False

    def connect(self):
        """
        Establish logical readiness by requiring a valid MCU ping response.
        """
        try:
            self.ping()
        except Exception as exc:
            self._connected = False
            raise ConnectionError(f"Failed to connect to UNO Q Bridge MCU endpoint: {exc}") from exc
        self._connected = True
        self._start_heartbeat()

    def disconnect(self):
        """
        Stop heartbeat tracking.  UNO Q Bridge transport has no raw port to close.
        """
        self._stop_heartbeat()
        self._heartbeat_healthy = False
        self._connected = False

    def _start_heartbeat(self):
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_failures = 0
        self._heartbeat_healthy = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="bridge-heartbeat",
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._heartbeat_stop.set()
        if (
            self._heartbeat_thread is not None
            and self._heartbeat_thread.is_alive()
            and self._heartbeat_thread is not threading.current_thread()
        ):
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.is_set():
            self._heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS)
            if self._heartbeat_stop.is_set():
                break
            try:
                self.ping()
                if self._heartbeat_failures > 0:
                    logger.info("Heartbeat recovered after %d failure(s)", self._heartbeat_failures)
                self._heartbeat_failures = 0
                self._heartbeat_healthy = True
            except Exception as exc:
                self._heartbeat_failures += 1
                if self._heartbeat_failures == 1:
                    logger.warning("Heartbeat failed: %s", exc)
                elif self._heartbeat_failures >= _HEARTBEAT_MAX_FAILURES:
                    if self._heartbeat_healthy:
                        logger.warning(
                            "Heartbeat failed %d consecutive times; marking connection unhealthy",
                            self._heartbeat_failures,
                        )
                        self._heartbeat_healthy = False

    def _request(self, command: Dict[str, Any], ack_timeout: float = 1.0) -> Dict[str, Any]:
        """
        Send one command via UNO Q Bridge and validate the returned ACK shape.

        ack_timeout is forwarded to the installed App Lab Bridge implementation.
        """
        endpoint, args = self._command_to_endpoint(command)
        with self._lock:
            response = self.transport.call(endpoint, *args, timeout=ack_timeout)
        response = self._normalize_response(command, response)

        if response.get("type") == "error":
            error_code = response.get("error")
            if not isinstance(error_code, str):
                raise RPCProtocolError(command, response, "error response missing 'error' key")
            raise RPCError(command, error_code, response)

        if not self._is_valid_ack(command, response):
            raise RPCProtocolError(command, response, "response is not a valid ACK for command")
        return response

    def _command_to_endpoint(self, command: Dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
        cmd_type = command.get("cmd")
        if cmd_type == "ping":
            return _ENDPOINTS["ping"], ()
        if cmd_type == "motor":
            return _ENDPOINTS["motor"], (command.get("direction"), command.get("speed"))
        if cmd_type == "mode":
            return _ENDPOINTS["mode"], (command.get("value"),)
        if cmd_type == "safe_reset":
            return _ENDPOINTS["safe_reset"], ()
        if cmd_type == "control_tick":
            return _ENDPOINTS["control_tick"], ()
        if cmd_type == "status":
            return _ENDPOINTS["status"], ()
        if cmd_type == "ultrasonic_diag":
            return _ENDPOINTS["ultrasonic_diag"], ()
        if cmd_type == "led":
            return _ENDPOINTS["led"], (command.get("value"),)
        if cmd_type == "buzzer":
            return _ENDPOINTS["buzzer"], (command.get("value"),)
        raise ValueError(f"Unknown command type: {cmd_type}")

    def _normalize_response(self, command: Dict[str, Any], response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            try:
                decoded = json.loads(response)
            except json.JSONDecodeError:
                raise RPCProtocolError(command, response, "response string is not JSON")
            if not isinstance(decoded, dict):
                raise RPCProtocolError(command, response, "decoded response is not a JSON object")
            return decoded
        raise RPCProtocolError(command, response, "response is not a dict or JSON object string")

    def send_command(self, command: Dict[str, Any], wait_for_ack: bool = True, ack_timeout: float = 1.0) -> bool:
        if not wait_for_ack:
            self._request(command, ack_timeout)
            return True
        self._request(command, ack_timeout)
        return True

    def _wait_for_ack(self, sent_command: Dict[str, Any], timeout: float) -> bool:
        self._request(sent_command, timeout)
        return True

    def _is_valid_ack(self, sent_command: Dict[str, Any], response: Dict[str, Any]) -> bool:
        resp_type = response.get("type")
        if not isinstance(resp_type, str):
            return False
        if sent_command.get("cmd") == "status":
            return self._is_valid_status(response)
        if sent_command.get("cmd") == "ultrasonic_diag":
            return self._is_valid_ultrasonic_diag(response)
        if not (resp_type.endswith("_ack") or resp_type == "pong"):
            return False

        cmd_type = sent_command.get("cmd")
        if cmd_type == "led":
            return resp_type == "led_ack" and response.get("color") == sent_command.get("value")
        if cmd_type == "buzzer":
            return resp_type == "buzzer_ack" and response.get("state") == sent_command.get("value")
        if cmd_type == "motor":
            return (
                resp_type == "motor_ack"
                and response.get("direction") == sent_command.get("direction")
                and response.get("speed") == sent_command.get("speed")
            )
        if cmd_type == "ping":
            return resp_type == "pong"
        if cmd_type == "mode":
            return resp_type == "mode_ack" and response.get("mode") == sent_command.get("value")
        if cmd_type == "safe_reset":
            return (
                resp_type == "safe_reset_ack"
                and response.get("status") == "ok"
                and response.get("mode") == "manual"
            )
        if cmd_type == "control_tick":
            return resp_type == "control_tick_ack" and isinstance(response.get("motion_authorized"), bool)
        return False

    def _is_valid_status(self, response: Dict[str, Any]) -> bool:
        if response.get("type") != "status":
            return False
        if response.get("mode") not in ("auto", "manual"):
            return False
        if response.get("movement") not in ("stopped", "forward", "backward", "left", "right"):
            return False
        if type(response.get("motor_speed")) is not int:
            return False
        for key in ("motion_lease_active", "control_tick_fresh", "command_fresh", "warning_active"):
            if not isinstance(response.get(key), bool):
                return False
        if response.get("safety_state") not in ("safe", "motion_authorized", "warning"):
            return False
        distances = response.get("distances")
        if not isinstance(distances, dict):
            return False
        for key in ("front", "rear", "left", "right"):
            value = distances.get(key)
            if value is not None and type(value) not in (int, float):
                return False
        return True

    def _is_valid_ultrasonic_diag(self, response: Dict[str, Any]) -> bool:
        if response.get("type") != "ultrasonic_diag":
            return False
        if response.get("mode") != "manual":
            return False
        distances = response.get("distances")
        if not isinstance(distances, dict):
            return False
        for key in ("front", "rear", "left", "right"):
            value = distances.get(key)
            if value is not None and type(value) not in (int, float):
                return False
        return True

    def led_control(self, color: str):
        if color not in ["red", "off"]:
            raise ValueError("Invalid LED color. Use 'red' or 'off'")
        return self.send_command({"cmd": "led", "value": color})

    def buzzer_control(self, state: str):
        if state not in ["on", "off"]:
            raise ValueError("Invalid buzzer state. Use 'on' or 'off'")
        return self.send_command({"cmd": "buzzer", "value": state})

    def motor_control(self, direction: str, speed: int = 200):
        if direction not in ["forward", "backward", "left", "right", "stop"]:
            raise ValueError("Invalid direction. Use 'forward', 'backward', 'left', 'right', or 'stop'")
        if type(speed) is not int:
            raise ValueError("Speed must be an integer")
        if not (0 <= speed <= 255):
            raise ValueError("Speed must be between 0 and 255")
        return self.send_command({"cmd": "motor", "direction": direction, "speed": speed})

    def ping(self):
        return self.send_command({"cmd": "ping"})

    def mode_control(self, mode: str):
        if mode not in ["auto", "manual"]:
            raise ValueError("Invalid mode. Use 'auto' or 'manual'")
        return self.send_command({"cmd": "mode", "value": mode})

    def safe_reset(self) -> bool:
        self._request({"cmd": "safe_reset"}, ack_timeout=2.0)
        return True

    def control_tick(self) -> bool:
        response = self._request({"cmd": "control_tick"}, ack_timeout=1.0)
        return bool(response.get("motion_authorized"))

    def get_status(self) -> Dict[str, Any]:
        return self._request({"cmd": "status"}, ack_timeout=1.0)

    def ultrasonic_diagnostic(self) -> Dict[str, Any]:
        return self._request({"cmd": "ultrasonic_diag"}, ack_timeout=2.0)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        with BridgeRPC() as bridge:
            bridge.ping()
            logger.info("UNO Q Bridge ping succeeded")
    except Exception as exc:
        logger.error("Bridge test failed: %s", exc)
