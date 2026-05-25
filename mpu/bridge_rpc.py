import serial
import json
import time
from typing import Dict, Any
from mpu.config import DEFAULT_SERIAL_PORT, DEFAULT_BAUDRATE, DEFAULT_TIMEOUT


class BridgeRPC:
    def __init__(self, port: str = DEFAULT_SERIAL_PORT, baudrate: int = DEFAULT_BAUDRATE, timeout: float = DEFAULT_TIMEOUT):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to {self.port}: {e}")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send_command(self, command: Dict[str, Any], wait_for_ack: bool = True, ack_timeout: float = 1.0) -> bool:
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Serial connection not established")

        try:
            json_cmd = json.dumps(command) + "\n"
            self.ser.write(json_cmd.encode('utf-8'))

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
        if color not in ["red", "green", "off"]:
            raise ValueError("Invalid LED color. Use 'red', 'green', or 'off'")

        command = {"cmd": "led", "value": color}
        return self.send_command(command)

    def buzzer_control(self, state: str):
        if state not in ["on", "off"]:
            raise ValueError("Invalid buzzer state. Use 'on' or 'off'")

        command = {"cmd": "buzzer", "value": state}
        return self.send_command(command)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


if __name__ == "__main__":
    try:
        with BridgeRPC() as bridge:
            bridge.led_control("red")
            time.sleep(1)
            bridge.buzzer_control("on")
            time.sleep(1)
            bridge.buzzer_control("off")
            time.sleep(1)
            bridge.led_control("green")
            time.sleep(1)
            bridge.led_control("off")
            print("Commands sent successfully")
    except Exception as e:
        print(f"Error: {e}")
