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

- Arduino UNO Q
- Logitech C922 USB Camera
- HC-SR04P Ultrasonic Sensors (x4)
- L298N Motor Driver
- DC Geared Motors (x4)

## Software Stack

- **MCU (STM32U585)** — Arduino C++ (motor control, obstacle avoidance, alerts)
- **MPU (Qualcomm QRB2210)** — Python (Edge Impulse inference, camera, dashboard communication)
- **App Lab** — JavaScript (real-time dashboard)

## Project Structure

```
Invent the Future with Arduino UNO Q and App Lab/
├── arduino/
│   ├── main.ino
│   ├── motor.h
│   ├── ultrasonic.h
│   ├── alert.h
│   └── bridge_rpc.h
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
   - Open `arduino/main.ino` in Arduino IDE
   - Select Arduino UNO Q board
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