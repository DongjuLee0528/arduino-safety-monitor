# Helmet Detection Safety Monitoring Robot

A safety monitoring robot that detects whether workers are wearing helmets using AI and alerts in real-time.

Built for the **Invent the Future with Arduino UNO Q and App Lab** competition.

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

Training runs for up to 30 epochs with early stopping. The best model is saved to `mpu/ai/models/best_model.pth`.

Current best result: **94.04% validation accuracy**