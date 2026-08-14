# Helmet Detection Safety Monitoring Robot

<!-- Project Badge Section -->
![Status](https://img.shields.io/badge/Status-Active-green)
![Platform](https://img.shields.io/badge/Platform-Arduino_UNO_Q-blue)
![Language](https://img.shields.io/badge/Language-C%2B%2B%2FPython-orange)
![AI](https://img.shields.io/badge/AI-EfficientNet_B0-purple)

A safety monitoring robot system that combines computer vision, autonomous navigation, and real-time alerting to detect worker helmet compliance in industrial environments using AI.

**Built for the Invent the Future with Arduino UNO Q and App Lab competition.**

## Project Mission

This project addresses critical workplace safety concerns by providing automated, real-time helmet detection and monitoring. The system helps prevent workplace accidents by ensuring safety protocol compliance through intelligent monitoring and immediate alert systems.

## Overview

An autonomous robot that patrols a workspace, detects helmet compliance using AI, and transmits real-time alerts to a monitoring server.

## Hardware

| Component | Quantity | Purpose |
|-----------|----------|---------|
| Arduino UNO Q | 1 | MCU — motor control, sensor reading, serial command processing |
| L298N Motor Driver | 1 | Dual H-bridge for TT motor control |
| DC Geared TT Motor | 4 | Drive wheels |
| HC-SR04 Ultrasonic Sensor | 4 | Obstacle detection (front, rear, left, right) |
| USB Camera | 1 | Video capture for AI inference |
| TXS0108E | 1 | 5V–3.3V logic level shifter |
| XL4015 | 1 | Step-down DC-DC converter |
| LED | 1 | Alert indicator (hardware implementation planned) |
| 5V Active Buzzer | 4 | Audio alert (hardware implementation planned) |

## Software Stack

| Layer | Language | Responsibility |
|-------|----------|----------------|
| Arduino UNO Q firmware | C++ | Motor control, ultrasonic sensing, JSON-RPC command processing |
| Python runtime (MPU) | Python | AI inference, camera capture, alert management, BridgeRPC |
| App Lab dashboard | JavaScript | Real-time monitoring dashboard (**planned — not yet implemented**) |

## Communication Architecture

**Transport: USB Serial JSON-RPC only. Wi-Fi and TCP are not used.**

```
Python Runtime (MPU)
        │
        │  BridgeRPC
        │  JSON commands over USB Serial (115200 baud)
        ▼
Arduino UNO Q
        │
        ▼
CommunicationManager    ← parses JSON, validates commands, detects timeout
        │
        ▼
RobotController         ← owns robot behaviour; reads mode and pending commands
        │
        ├─► NavigationManager   ← autonomous navigation state machine
        │
        └─► MotorController     ← L298N PWM motor control

App Lab Dashboard  ← FUTURE / PLANNED — not yet implemented
```

### JSON Command Protocol

Each command is a single JSON object terminated by `\n`.
Each response is a single JSON object terminated by `\n`.

| Command | Allowed in AUTO | Effect |
|---------|-----------------|--------|
| `{"cmd":"motor","direction":"forward","speed":<0-255>}` | No | Queue forward movement |
| `{"cmd":"motor","direction":"backward","speed":<0-255>}` | No | Queue backward movement |
| `{"cmd":"motor","direction":"left","speed":<0-255>}` | No | Queue left turn |
| `{"cmd":"motor","direction":"right","speed":<0-255>}` | No | Queue right turn |
| `{"cmd":"motor","direction":"stop"}` | No | Stop motors |
| `{"cmd":"mode","value":"auto"}` | — | Switch to autonomous mode |
| `{"cmd":"mode","value":"manual"}` | — | Switch to manual mode; stop motors |
| `{"cmd":"ping"}` | — | Connectivity test; responds `{"type":"pong"}` |
| `{"cmd":"safe_reset"}` | — | Stop motors, force MANUAL mode |

Speed must be a strict JSON integer in the range 0–255. Floats, booleans, strings, and null are rejected with `{"type":"error","error":"INVALID_SPEED"}`.

### Timeout and Safety Behaviour

- If no byte is received from the host within `SERIAL_CMD_TIMEOUT_MS` (default 2000 ms), the connection is considered lost.
- On timeout: motors stop immediately; mode switches to MANUAL.
- The robot resumes only after a new valid command is received.
- `safe_reset` always stops motors and forces MANUAL mode regardless of current state.

## Pin Map

### L298N Motor Driver

| Signal | Pin | Notes |
|--------|-----|-------|
| ENA (left channel PWM) | D5 | PWM-capable |
| IN1 (left direction A) | D2 | |
| IN2 (left direction B) | D4 | |
| ENB (right channel PWM) | D6 | PWM-capable |
| IN3 (right direction A) | D7 | |
| IN4 (right direction B) | D8 | |

### HC-SR04 Ultrasonic Sensors

| Sensor | TRIG | ECHO |
|--------|------|------|
| Front | D9 | D10 |
| Rear | D11 | D12 |
| Left | A0 | A1 |
| Right | A2 | A3 |

A0–A3 are used as digital GPIO; no ADC reads occur on these pins.

### Reserved Pins

| Pin | Reason |
|-----|--------|
| D0 | UART RX |
| D1 | UART TX |
| D3 | Spare PWM |
| D13 | Built-in LED |
| A4 | I2C SDA |
| A5 | I2C SCL |

## Implemented Features

- **Arduino UNO Q** — C++ firmware with no Wi-Fi dependency
- **BridgeRPC** — JSON-RPC over USB Serial; strict integer speed validation
- **RobotController** — top-level orchestrator; AUTO and MANUAL modes
- **NavigationManager** — autonomous obstacle-avoidance state machine
- **MotorController** — L298N PWM speed and direction control
- **UltrasonicSensor** — non-blocking round-robin HC-SR04 readings with rolling average
- **AUTO mode** — autonomous forward patrol with obstacle avoidance
- **MANUAL mode** — remote command-driven movement
- **Safety stops** — timeout detection, safe_reset, connection-loss handling
- **DashboardState** — in-process state mirror (connection, detection, statistics, events)
- **Daily Statistics** — per-UTC-day worker inspection counts (inspected, helmet, no-helmet)
- **Runtime Event Logging** — timestamped event queue (connection, detection, alert, system)
- **Person Detection** — MobileNet SSD person detector
- **Helmet Classification** — EfficientNet-B0 helmet/no-helmet classifier
- **Alert Transmission** — HTTP POST with base64 image, UTC timestamp, detection metadata

## Future Work

The following items are planned and not yet implemented:

- **Arduino App Lab UI** — App Lab JavaScript dashboard for real-time robot monitoring and control
- **App Lab integration** — connecting the App Lab dashboard to the Python runtime over HTTP or WebSocket
- **LED hardware implementation** — wiring and firmware for the alert LED indicator; the protocol command exists but the GPIO implementation is not complete
- **Four 5V active buzzer hardware implementation** — wiring and firmware for audio alerts; the protocol command exists but the GPIO implementation is not complete

## Project Structure

```
Invent the Future with Arduino UNO Q and App Lab/
├── README.md
├── requirements.txt
├── setup.cfg
├── arduino/
│   ├── arduino.ino          ← entry point; setup() and main loop
│   ├── config.h             ← all tuneable constants (speeds, timeouts, baud rate)
│   ├── pins.h               ← all GPIO pin assignments
│   ├── motor.h              ← MotorController (L298N)
│   ├── ultrasonic.h         ← UltrasonicSensor (HC-SR04 ×4, non-blocking)
│   ├── navigation.h         ← NavigationManager (autonomous state machine)
│   ├── comm.h               ← CommunicationManager (USB Serial JSON-RPC)
│   └── robot_controller.h   ← RobotController (top-level orchestrator)
└── mpu/
    ├── __init__.py
    ├── main.py              ← HelmetDetectionSystem; main loop
    ├── camera.py            ← CameraCapture
    ├── classifier.py        ← HelmetClassifier (EfficientNet-B0)
    ├── detector.py          ← PersonDetector (MobileNet SSD)
    ├── alert_manager.py     ← AlertManager (cooldown logic)
    ├── bridge_rpc.py        ← BridgeRPC (USB Serial JSON-RPC client)
    ├── sender.py            ← Sender (HTTP alert transmission)
    ├── config.py            ← runtime configuration constants
    ├── dashboard_state.py   ← DashboardState (in-process state mirror)
    ├── tests/
    │   ├── test_bridge_rpc.py
    │   ├── test_daily_statistics.py
    │   ├── test_dashboard_state.py
    │   ├── test_event_logging.py
    │   ├── test_main_integration.py
    │   └── test_phase2_fixes.py
    └── ai/
        ├── train.py
        ├── convert.py
        └── dataset/
            └── loader.py
```

## Datasets

- **SHEL5K** — 5,000 images, Pascal VOC XML ([Mendeley Data](https://data.mendeley.com/datasets/9rcv8mm682/4))
- **SHWD** — 7,581 images, Pascal VOC XML ([GitHub](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset))

## Training

Download the datasets and place them as follows:

```
~/Documents/AIdatasets/helmet-safety-robot/raw/
├── 9rcv8mm682-4/Safety Helmet Wearing Dataset/
└── VOC2028/
```

Then run:

```bash
python -m mpu.ai.train
```

### Training Configuration

- **Architecture**: EfficientNet-B0 with custom classification head
- **Transfer Learning**: ImageNet pre-trained weights
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropy for binary classification
- **Early Stopping**: Patience of 5 epochs based on validation loss
- **Regularization**: Dropout (0.2) in classification layer

Training runs for up to 30 epochs with early stopping. The best model is saved to `mpu/ai/models/best_model.pth`.

### Model Performance

- **Current Best Result**: **94.04% validation accuracy**
- **Model Size**: ~17 MB (PyTorch), ~14 MB (ONNX optimized)
- **Inference Speed**: ~50 ms per frame on typical hardware
- **Deployment Format**: ONNX for cross-platform compatibility

## Quick Start

### Prerequisites

- Arduino UNO Q with hardware connected per the pin map above
- Python 3.8+
- USB camera (Logitech C922 or compatible)
- Datasets downloaded and positioned correctly

### Installation

1. **Clone repository**:
   ```bash
   git clone [repository-url]
   cd "Invent the Future with Arduino UNO Q and App Lab"
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Upload Arduino firmware**:
   - Open `arduino/arduino.ino` in Arduino IDE
   - Select **Arduino UNO Q** board
   - Upload to device via USB

4. **Run the system in App Lab hardware mode**:
   ```bash
   APP_LAB_DEV_MODE=false python -m mpu.main --server-url http://localhost:3000/api/alert
   ```

## Configuration

### Arduino

- **Pin assignments**: `arduino/pins.h`
- **Motor speeds and timeouts**: `arduino/config.h`
- **Obstacle threshold**: `OBSTACLE_THRESHOLD_CM` in `arduino/config.h`

### Python

- **MCU transport**: UNO Q App Lab Bridge IPC (`DEFAULT_SERIAL_PORT` is a compatibility label)
- **Camera resolution**: `CAMERA_WIDTH`, `CAMERA_HEIGHT` in `mpu/config.py`
- **Alert server URL**: `DEFAULT_SERVER_URL` in `mpu/config.py`

## Testing

```bash
python -m unittest
```

Tests cover BridgeRPC protocol, DashboardState, daily statistics, event logging, main pipeline integration, detector/classifier validation, and MCU safety behavior.

## Troubleshooting

### Camera not detected

```bash
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

### UNO Q Bridge unavailable

```bash
python -c "from arduino.app_utils import Bridge; print(Bridge.call('asm_ping'))"
```

### Model file missing

```bash
python -m mpu.ai.train
```

## Competition Details

**Competition**: Invent the Future with Arduino UNO Q and App Lab
**Category**: Safety and Security Solutions
**Technology Stack**: Arduino UNO Q, Python, AI/ML
**Innovation Focus**: Workplace safety automation through AI-powered monitoring

---

**Built for workplace safety and AI innovation**
