# Helmet Detection Safety Monitoring Robot

<!-- Project Badge Section -->
![Status](https://img.shields.io/badge/Status-Active-green)
![Platform](https://img.shields.io/badge/Platform-Arduino_UNO_Q-blue)
![Language](https://img.shields.io/badge/Language-C%2B%2B%2FPython-orange)
![AI](https://img.shields.io/badge/AI-EfficientNet_B0-purple)

A comprehensive safety monitoring robot system that combines computer vision, autonomous navigation, and real-time alerting to detect worker helmet compliance in industrial environments using advanced AI technology.

**Built for the Invent the Future with Arduino UNO Q and App Lab competition.**

## 🎯 Project Mission

This project addresses critical workplace safety concerns by providing automated, real-time helmet detection and monitoring. The system helps prevent workplace accidents by ensuring safety protocol compliance through intelligent monitoring and immediate alert systems.

## Overview

An autonomous robot that patrols a workspace, detects helmet compliance using Edge Impulse AI, and sends real-time alerts via an App Lab dashboard.

## Hardware

- Arduino UNO R4 WiFi
- Logitech C922 USB Camera
- HC-SR04 Ultrasonic Sensors (x4)
- L298N Motor Driver
- DC Geared TT Motors (x4)

## Software Stack

- **Arduino UNO R4 WiFi** — Arduino C++ (motor control, obstacle avoidance, Wi-Fi communication)
- **MPU** — Python (AI inference, camera, dashboard communication)
- **App Lab** — JavaScript (real-time dashboard)

## Communication Architecture

**Transport: Wi-Fi TCP only. Bluetooth is not used.**

```
App Lab (Mac)
    │
    │  Wi-Fi TCP  (port WIFI_SERVER_PORT, plain-text commands)
    ▼
Arduino UNO R4 WiFi
    │
    ▼
CommunicationManager   ← receives commands, validates, detects timeout/loss
    │
    ▼
RobotController        ← owns robot behaviour, reads mode and pending commands
    │
    ├─► NavigationManager   ← owns autonomous navigation decisions
    │
    └─► MotorController     ← owns motor control
```

### Command Set

| Command    | Allowed in AUTO | Effect |
|------------|-----------------|--------|
| `FORWARD`  | No              | Queue forward movement |
| `BACKWARD` | No              | Queue backward movement |
| `LEFT`     | No              | Queue left turn |
| `RIGHT`    | No              | Queue right turn |
| `STOP`     | Yes             | Immediate stop; mode → MANUAL |
| `AUTO`     | —               | Switch to autonomous mode |
| `MANUAL`   | —               | Switch to manual mode; stop motors |

Commands are plain ASCII text, newline-terminated, case-insensitive. No JSON, no binary protocol.

### Timeout and Safety Behaviour

- If the Wi-Fi client sends no data within `WIFI_CMD_TIMEOUT_MS` (default 2000 ms), the connection is considered lost.
- On connection loss or timeout: motors stop immediately; mode switches to MANUAL.
- The robot does **not** resume automatically. It resumes only after a new Wi-Fi connection is established and a valid command is received.
- `STOP` always overrides any movement, regardless of mode.

### USB Serial

USB Serial (`SERIAL_BAUD_RATE` = 115200) is used for debug diagnostics and development only. It is **not** the primary runtime control interface.

## Project Structure

```
Invent the Future with Arduino UNO Q and App Lab/
├── arduino/
│   ├── arduino.ino          ← entry point; Wi-Fi init + main loop
│   ├── config.h             ← all tuneable constants (speeds, timeouts, Wi-Fi placeholders)
│   ├── pins.h               ← all GPIO pin assignments
│   ├── motor.h              ← MotorController (L298N)
│   ├── ultrasonic.h         ← UltrasonicSensor (HC-SR04 ×4)
│   ├── navigation.h         ← NavigationManager (autonomous state machine)
│   ├── comm.h               ← CommunicationManager (Wi-Fi TCP)
│   └── robot_controller.h  ← RobotController (top-level orchestrator)
└── mpu/
    ├── main.py
    ├── camera.py
    ├── classifier.py
    ├── alert_manager.py
    ├── bridge_rpc.py
    ├── sender.py
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

### 📈 Training Configuration

- **Architecture**: EfficientNet-B0 with custom classification head
- **Transfer Learning**: ImageNet pre-trained weights
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropy for binary classification
- **Early Stopping**: Patience of 5 epochs based on validation loss
- **Regularization**: Dropout (0.2) in classification layer

Training runs for up to 30 epochs with early stopping. The best model is saved to `mpu/ai/models/best_model.pth`.

### 🎯 Model Performance

- **Current Best Result**: **94.04% validation accuracy**
- **Model Size**: ~17MB (PyTorch), ~14MB (ONNX optimized)
- **Inference Speed**: ~50ms per frame on typical hardware
- **Deployment Format**: ONNX for cross-platform compatibility

## 🚀 Quick Start Guide

### Prerequisites
- Arduino UNO Q with properly connected hardware
- Python 3.8+ environment
- USB camera (Logitech C922 or compatible)
- Datasets downloaded and positioned correctly

### Installation Steps

1. **Clone Repository**:
   ```bash
   git clone [repository-url]
   cd "Invent the Future with Arduino UNO Q and App Lab"
   ```

2. **Setup Python Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Upload Arduino Code**:
   - Open `arduino/arduino.ino` in Arduino IDE
   - Select **Arduino UNO R4 WiFi** board
   - Set Wi-Fi credentials in `arduino/config.h` (`WIFI_SSID`, `WIFI_PASSWORD`)
   - Upload to device

4. **Configure System**:
   ```bash
   # Edit configuration if needed
   nano mpu/config.py
   ```

5. **Run System**:
   ```bash
   # Start the helmet detection system
   python -m mpu.main --port /dev/ttyUSB0 --server-url http://localhost:3000/api/alert
   ```

## 🔧 Configuration Options

### Arduino Configuration
- **Pin Assignments**: Modify `arduino/pins.h` for custom hardware setup
- **Motor Parameters**: Adjust speeds and timing in `motor.h`
- **Sensor Thresholds**: Configure obstacle detection distances in `ultrasonic.h`

### Python Configuration
- **Serial Port**: Update `DEFAULT_SERIAL_PORT` in `config.py`
- **Camera Settings**: Modify `CAMERA_WIDTH` and `CAMERA_HEIGHT`
- **Alert Thresholds**: Adjust `DEFAULT_DETECTION_THRESHOLD` and `DEFAULT_COOLDOWN_TIME`
- **Server URL**: Configure `DEFAULT_SERVER_URL` for dashboard communication

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

#### Serial Communication Errors
```bash
# Check available serial ports
python -c "import serial.tools.list_ports; [print(p) for p in serial.tools.list_ports.comports()]"
```

#### Model File Missing
```bash
# Train a new model or download pre-trained
python -m mpu.ai.train
```

### Performance Optimization
- **Inference Speed**: Use ONNX model instead of PyTorch for faster inference
- **Memory Usage**: Reduce camera resolution if experiencing memory issues
- **Detection Accuracy**: Increase `DEFAULT_DETECTION_THRESHOLD` to reduce false positives

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow existing conventions and add comprehensive comments
2. **Testing**: Test all changes with actual hardware before submitting
3. **Documentation**: Update README.md and inline documentation for new features
4. **Commit Messages**: Use clear, descriptive commit messages

## 📄 License

This project is developed for the "Invent the Future with Arduino UNO Q and App Lab" competition.

## 🏆 Competition Details

**Competition**: Invent the Future with Arduino UNO Q and App Lab
**Category**: Safety and Security Solutions
**Technology Stack**: Arduino UNO Q, Python, AI/ML, App Lab Dashboard
**Innovation Focus**: Workplace safety automation through AI-powered monitoring

---

**Built with ❤️ for workplace safety and AI innovation**