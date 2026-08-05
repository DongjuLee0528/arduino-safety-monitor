/*
 * arduino.ino – Arduino UNO Q Main Sketch
 *
 * Entry point.  Constructs all subsystem objects and delegates every loop
 * iteration to CommunicationManager and RobotController.
 *
 * Communication transport: USB Serial (BridgeRPC JSON protocol).
 * This is the sole active communication path.
 *
 * Module dependency graph (arrows = "uses"):
 *
 *   arduino.ino
 *     ├─► CommunicationManager  (Serial JSON-RPC transport, command parsing, state)
 *     └─► RobotController       (per-loop orchestrator, all motor actions)
 *           ├─► MotorController
 *           ├─► UltrasonicSensor
 *           ├─► NavigationManager
 *           └─► CommunicationManager
 *
 * setup() sequence:
 *   1. Serial.begin()   – primary communication and debug port
 *   2. motor.begin()    – configure motor pins, enter stopped state
 *   3. comm.begin()     – emit ready signal over Serial
 *
 * No alert, LED, or buzzer objects exist in this sketch.
 * No Wi-Fi.  No network credentials.
 */

#include "config.h"
#include "pins.h"
#include "motor.h"
#include "ultrasonic.h"
#include "navigation.h"
#include "comm.h"
#include "robot_controller.h"

// Left: IN1/IN2 direction pins, right: IN1/IN2 direction pins, then PWM enable pins
MotorController motor(
    MOTOR_LEFT_IN1_PIN,    MOTOR_LEFT_IN2_PIN,
    MOTOR_RIGHT_IN1_PIN,   MOTOR_RIGHT_IN2_PIN,
    MOTOR_LEFT_ENABLE_PIN, MOTOR_RIGHT_ENABLE_PIN
);

// Trigger/echo pin pairs for front, rear, left, and right HC-SR04 sensors
UltrasonicSensor ultrasonic(
    ULTRASONIC_FRONT_TRIGGER_PIN, ULTRASONIC_FRONT_ECHO_PIN,
    ULTRASONIC_REAR_TRIGGER_PIN,  ULTRASONIC_REAR_ECHO_PIN,
    ULTRASONIC_LEFT_TRIGGER_PIN,  ULTRASONIC_LEFT_ECHO_PIN,
    ULTRASONIC_RIGHT_TRIGGER_PIN, ULTRASONIC_RIGHT_ECHO_PIN
);

CommunicationManager comm;                       // Handles Serial JSON-RPC transport
RobotController      robot(&motor, &ultrasonic, &comm); // Orchestrates all motion logic

void setup() {
    Serial.begin(SERIAL_BAUD_RATE); // Open serial port for JSON-RPC communication
    motor.begin();                  // Configure motor GPIO pins and enter stopped state
    ultrasonic.begin();             // Configure ultrasonic sensor GPIO pins
    comm.begin();                   // Emit ready signal so the host knows the MCU is up
}

void loop() {
    comm.update();          // Parse incoming JSON commands and update internal state
    robot.update();         // Execute one control cycle (obstacle check, motor commands)
    delay(LOOP_DELAY_MS);   // Fixed loop rate defined in config.h
}
