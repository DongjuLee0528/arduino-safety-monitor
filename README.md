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