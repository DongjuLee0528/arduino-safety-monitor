import serial
import json
import time
from typing import Dict, Any


class BridgeRPC:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout: float = 1.0):
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

    def send_command(self, command: Dict[str, Any]) -> bool:
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Serial connection not established")

        try:
            json_cmd = json.dumps(command) + "\n"
            self.ser.write(json_cmd.encode('utf-8'))
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to send command: {e}")

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
