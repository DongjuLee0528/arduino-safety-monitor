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

MotorController motor(
    MOTOR_LEFT_IN1_PIN,    MOTOR_LEFT_IN2_PIN,
    MOTOR_RIGHT_IN1_PIN,   MOTOR_RIGHT_IN2_PIN,
    MOTOR_LEFT_ENABLE_PIN, MOTOR_RIGHT_ENABLE_PIN
);

UltrasonicSensor ultrasonic(
    ULTRASONIC_FRONT_TRIGGER_PIN, ULTRASONIC_FRONT_ECHO_PIN,
    ULTRASONIC_REAR_TRIGGER_PIN,  ULTRASONIC_REAR_ECHO_PIN,
    ULTRASONIC_LEFT_TRIGGER_PIN,  ULTRASONIC_LEFT_ECHO_PIN,
    ULTRASONIC_RIGHT_TRIGGER_PIN, ULTRASONIC_RIGHT_ECHO_PIN
);

CommunicationManager comm;
RobotController      robot(&motor, &ultrasonic, &comm);

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    motor.begin();
    comm.begin();
}

void loop() {
    comm.update();
    robot.update();
    delay(LOOP_DELAY_MS);
}
