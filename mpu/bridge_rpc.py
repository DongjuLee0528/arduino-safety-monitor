"""
Bridge RPC Communication Module

This module provides a Python interface for communicating with the Arduino-based
robot system via JSON-RPC over serial communication. It handles command transmission,
acknowledgment verification, and connection management.

Key Features:
- JSON-based command protocol for LED, buzzer, and motor control
- Automatic acknowledgment verification with timeout handling
- Context manager support for automatic connection management
- Robust error handling and connection recovery

Supported Commands:
- LED control: {"cmd": "led", "value": "red"/"off"}
- Buzzer control: {"cmd": "buzzer", "value": "on"/"off"}
- Motor control: {"cmd": "motor", "direction": "forward"/"backward"/"left"/"right"/"stop", "speed": 0-255}
- Ping: {"cmd": "ping"} for connectivity testing

Usage:
    with BridgeRPC("/dev/ttyUSB0") as bridge:
        bridge.led_control("red")
        bridge.buzzer_control("on")
"""

import serial
import json
import time
import logging
import threading
from typing import Dict, Any, Optional
from mpu.config import DEFAULT_SERIAL_PORT, DEFAULT_BAUDRATE, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 0.5  # Seconds between ping commands sent by the heartbeat thread
_HEARTBEAT_MAX_FAILURES = 3       # Consecutive ping failures before the connection is marked unhealthy

# Message types sent asynchronously by the Arduino; not ACKs and should be silently skipped
_ASYNC_MESSAGE_TYPES = frozenset({"sensor_data", "ultrasonic", "motor_status", "avoidance"})
# All valid ACK type strings expected in responses to commands
_ACK_TYPES = frozenset({"pong", "motor_ack", "led_ack", "buzzer_ack", "mode_ack", "safe_reset_ack", "control_tick_ack"})


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


class BridgeRPC:
    """
    JSON-RPC communication bridge for Arduino robot control.

    This class manages serial communication with the Arduino, providing
    high-level methods for controlling robot components and verifying
    command execution through acknowledgment messages.
    """
    def __init__(self, port: str = DEFAULT_SERIAL_PORT, baudrate: int = DEFAULT_BAUDRATE, timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize the bridge RPC communication interface.

        Args:
            port: Serial port device path (e.g., '/dev/ttyUSB0')
            baudrate: Communication speed in bits per second
            timeout: Read/write timeout in seconds
        """
        self.port = port            # Serial port device path
        self.baudrate = baudrate    # Communication baud rate
        self.timeout = timeout      # Read/write timeout
        self.ser = None             # Serial connection object (initialized in connect())

        self._lock = threading.Lock()                          # Serialises serial write+read pairs
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()               # Set to signal the heartbeat thread to exit
        self._heartbeat_failures = 0                           # Consecutive ping failures since last recovery
        self._heartbeat_healthy = True                         # False after _HEARTBEAT_MAX_FAILURES failures

    def connect(self):
        """
        Establish serial connection to Arduino.

        Raises:
            ConnectionError: If serial connection fails
        """
        try:
            # Open serial connection with specified parameters
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Allow Arduino to reset and stabilize
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to {self.port}: {e}")

        self._start_heartbeat()

    def disconnect(self):
        """
        Close the serial connection if it's open.
        """
        self._stop_heartbeat()
        self._heartbeat_healthy = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _start_heartbeat(self):
        # Do nothing if a heartbeat thread is already running
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_failures = 0
        self._heartbeat_healthy = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,      # Thread exits automatically when the main process ends
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
            if not self.ser or not self.ser.is_open:
                break
            try:
                self._request({"cmd": "ping"}, ack_timeout=1.0)
                if self._heartbeat_failures > 0:
                    logger.info("Heartbeat recovered after %d failure(s)", self._heartbeat_failures)
                self._heartbeat_failures = 0
                self._heartbeat_healthy = True
            except Exception as e:
                self._heartbeat_failures += 1
                if self._heartbeat_failures == 1:
                    logger.warning("Heartbeat failed: %s", e)
                elif self._heartbeat_failures >= _HEARTBEAT_MAX_FAILURES:
                    if self._heartbeat_healthy:
                        logger.warning(
                            "Heartbeat failed %d consecutive times; marking connection unhealthy",
                            self._heartbeat_failures,
                        )
                        self._heartbeat_healthy = False

    def _request(self, command: Dict[str, Any], ack_timeout: float = 1.0) -> Dict[str, Any]:
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Serial connection not established")

        with self._lock:
            try:
                json_cmd = json.dumps(command) + "\n"
                self.ser.write(json_cmd.encode("utf-8"))
            except Exception as e:
                raise RuntimeError(f"Failed to send command: {e}")

            return self._read_response(command, ack_timeout)

    def _read_response(self, command: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        buf = ""  # Accumulates partial data between reads

        while time.monotonic() < deadline:
            if self.ser.in_waiting > 0:
                buf += self.ser.read(self.ser.in_waiting).decode("utf-8")
                lines = buf.split("\n")
                buf = lines[-1]  # Keep any incomplete trailing line for the next iteration
                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        response = json.loads(line)
                    except json.JSONDecodeError:
                        raise RPCProtocolError(command, line, "JSON parse failure")

                    if not isinstance(response, dict):
                        raise RPCProtocolError(command, response, "response is not a JSON object")

                    resp_type = response.get("type")

                    if resp_type == "error":  # Arduino explicitly reported an error
                        error_code = response.get("error")
                        if not isinstance(error_code, str):
                            raise RPCProtocolError(command, response, "error response missing 'error' key")
                        raise RPCError(command, error_code, response)

                    if resp_type is None:
                        continue  # Ignore messages with no type field

                    if resp_type in _ASYNC_MESSAGE_TYPES:
                        continue  # Silently discard unsolicited telemetry messages

                    if self._is_valid_ack(command, response):
                        return response  # Correct ACK received

                    raise RPCProtocolError(
                        command,
                        response,
                        f"expected ACK for '{command.get('cmd')}' but got type '{resp_type}'",
                    )
            time.sleep(0.01)  # Yield CPU while waiting for incoming bytes

        raise TimeoutError(
            f"No valid ACK received within {timeout}s for command: {command}"
        )

    def send_command(self, command: Dict[str, Any], wait_for_ack: bool = True, ack_timeout: float = 1.0) -> bool:
        """
        Send JSON command to Arduino and optionally wait for acknowledgment.

        Args:
            command: Dictionary containing command data
            wait_for_ack: Whether to wait for acknowledgment response
            ack_timeout: Maximum time to wait for acknowledgment

        Returns:
            True if command sent successfully (and acknowledged if requested)

        Raises:
            ConnectionError: If serial connection is not established
            RuntimeError: If command transmission fails
            TimeoutError: If acknowledgment not received within timeout
        """
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Serial connection not established")

        if not wait_for_ack:
            with self._lock:
                try:
                    # Convert command to JSON and send via serial
                    json_cmd = json.dumps(command) + "\n"
                    self.ser.write(json_cmd.encode("utf-8"))
                    return True
                except Exception as e:
                    raise RuntimeError(f"Failed to send command: {e}")

        # Wait for acknowledgment if requested
        self._request(command, ack_timeout)
        return True

    def _wait_for_ack(self, sent_command: Dict[str, Any], timeout: float) -> bool:
        self._read_response(sent_command, timeout)
        return True

    def _is_valid_ack(self, sent_command: Dict[str, Any], response: Dict[str, Any]) -> bool:
        resp_type = response.get("type")
        if not isinstance(resp_type, str):
            return False
        if not (resp_type.endswith("_ack") or resp_type == "pong"):
            return False

        cmd_type = sent_command.get("cmd")
        if cmd_type == "led":
            return (resp_type == "led_ack" and
                   response.get("color") == sent_command.get("value"))
        elif cmd_type == "buzzer":
            return (resp_type == "buzzer_ack" and
                   response.get("state") == sent_command.get("value"))
        elif cmd_type == "motor":
            return (resp_type == "motor_ack" and
                   response.get("direction") == sent_command.get("direction") and
                   response.get("speed") == sent_command.get("speed"))
        elif cmd_type == "ping":
            return resp_type == "pong"
        elif cmd_type == "mode":
            return (resp_type == "mode_ack" and
                   response.get("mode") == sent_command.get("value"))
        elif cmd_type == "safe_reset":
            return (resp_type == "safe_reset_ack" and
                    response.get("status") == "ok" and
                    response.get("mode") == "manual")
        elif cmd_type == "control_tick":
            return (resp_type == "control_tick_ack" and
                    isinstance(response.get("motion_authorized"), bool))

        return False

    def led_control(self, color: str):
        """
        Control LED color on the Arduino.

        Args:
            color: LED color command ('red' or 'off')

        Returns:
            True if command executed successfully

        Raises:
            ValueError: If invalid color specified
        """
        if color not in ["red", "off"]:
            raise ValueError("Invalid LED color. Use 'red' or 'off'")

        command = {"cmd": "led", "value": color}
        return self.send_command(command)

    def buzzer_control(self, state: str):
        """
        Control buzzer state on the Arduino.

        Args:
            state: Buzzer state command ('on' or 'off')

        Returns:
            True if command executed successfully

        Raises:
            ValueError: If invalid state specified
        """
        if state not in ["on", "off"]:
            raise ValueError("Invalid buzzer state. Use 'on' or 'off'")

        command = {"cmd": "buzzer", "value": state}
        return self.send_command(command)

    def motor_control(self, direction: str, speed: int = 200):
        """
        Control motor movement on the Arduino.

        Args:
            direction: Movement direction ('forward', 'backward', 'left', 'right', 'stop')
            speed: PWM speed value (0-255), default 200

        Returns:
            True if command executed successfully

        Raises:
            ValueError: If invalid direction or speed specified
        """
        if direction not in ["forward", "backward", "left", "right", "stop"]:
            raise ValueError("Invalid direction. Use 'forward', 'backward', 'left', 'right', or 'stop'")

        if type(speed) is not int:
            raise ValueError("Speed must be an integer")
        if not (0 <= speed <= 255):
            raise ValueError("Speed must be between 0 and 255")

        command = {"cmd": "motor", "direction": direction, "speed": speed}
        return self.send_command(command)

    def ping(self):
        """
        Send ping command to Arduino for connectivity testing.

        Returns:
            True if pong response received successfully

        Raises:
            TimeoutError: If no pong response received
        """
        command = {"cmd": "ping"}
        return self.send_command(command)

    def mode_control(self, mode: str):
        """
        Set the operating mode on the Arduino.

        Args:
            mode: Operating mode ('auto' or 'manual')

        Returns:
            True if command executed successfully

        Raises:
            ValueError: If invalid mode specified
        """
        if mode not in ["auto", "manual"]:
            raise ValueError("Invalid mode. Use 'auto' or 'manual'")

        command = {"cmd": "mode", "value": mode}
        return self.send_command(command)

    def safe_reset(self) -> bool:
        """Reset the MCU to manual mode and clear any active safe-mode state."""
        command = {"cmd": "safe_reset"}
        self._request(command, ack_timeout=2.0)  # Longer timeout; MCU may be resetting state
        return True

    def control_tick(self) -> bool:
        """
        Renew the motion lease on the MCU.

        Must be called only from the Python main control loop.
        Must NOT be called from the heartbeat thread.

        Returns:
            True if the MCU reports motion_authorized=true.
            False if the MCU reports motion_authorized=false.

        Raises:
            RPCError: If the Arduino returns an error response.
            RPCProtocolError: If the response is malformed.
            TimeoutError: If no ACK is received within 1 second.
            ConnectionError: If the serial connection is not open.
        """
        command = {"cmd": "control_tick"}
        response = self._request(command, ack_timeout=1.0)
        return bool(response.get("motion_authorized"))

    def __enter__(self):
        """
        Context manager entry - establish connection.
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - close connection.
        """
        self.disconnect()


if __name__ == "__main__":
    """
    Test script for BridgeRPC functionality.
    Demonstrates LED and buzzer control sequence.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        # Test Arduino communication using context manager
        with BridgeRPC() as bridge:
            logger.info("Testing Arduino communication...")

            # Test LED control
            bridge.led_control("red")
            time.sleep(1)

            # Test buzzer control
            bridge.buzzer_control("on")
            time.sleep(1)
            bridge.buzzer_control("off")
            time.sleep(1)

            # Test LED off
            bridge.led_control("off")
            time.sleep(1)

            logger.info("Commands sent successfully")
    except Exception as e:
        logger.error("Error: %s", e)
