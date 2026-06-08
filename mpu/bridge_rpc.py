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
from typing import Dict, Any
from mpu.config import DEFAULT_SERIAL_PORT, DEFAULT_BAUDRATE, DEFAULT_TIMEOUT


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

    def disconnect(self):
        """
        Close the serial connection if it's open.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()

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

        try:
            # Convert command to JSON and send via serial
            json_cmd = json.dumps(command) + "\n"
            self.ser.write(json_cmd.encode('utf-8'))

            # Wait for acknowledgment if requested
            if wait_for_ack:
                return self._wait_for_ack(command, ack_timeout)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to send command: {e}")

    def _wait_for_ack(self, sent_command: Dict[str, Any], timeout: float) -> bool:
        start_time = time.time()
        response_buffer = ""

        while time.time() - start_time < timeout:
            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting).decode('utf-8')
                response_buffer += data

                lines = response_buffer.split('\n')
                response_buffer = lines[-1]

                for line in lines[:-1]:
                    if line.strip():
                        try:
                            response = json.loads(line.strip())
                            if self._is_valid_ack(sent_command, response):
                                return True
                        except json.JSONDecodeError:
                            continue
            time.sleep(0.01)

        raise TimeoutError(f"No valid ACK received within {timeout} seconds for command: {sent_command}")

    def _is_valid_ack(self, sent_command: Dict[str, Any], response: Dict[str, Any]) -> bool:
        if not response.get("type", "").endswith("_ack"):
            return False

        cmd_type = sent_command.get("cmd")
        if cmd_type == "led":
            return (response.get("type") == "led_ack" and
                   response.get("color") == sent_command.get("value"))
        elif cmd_type == "buzzer":
            return (response.get("type") == "buzzer_ack" and
                   response.get("state") == sent_command.get("value"))
        elif cmd_type == "motor":
            return (response.get("type") == "motor_ack" and
                   response.get("direction") == sent_command.get("direction") and
                   response.get("speed") == sent_command.get("speed"))
        elif cmd_type == "ping":
            return response.get("type") == "pong"

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
    try:
        # Test Arduino communication using context manager
        with BridgeRPC() as bridge:
            print("Testing Arduino communication...")

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

            print("Commands sent successfully")
    except Exception as e:
        print(f"Error: {e}")
