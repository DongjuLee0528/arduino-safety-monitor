/*
 * arduino.ino – Arduino UNO Q Main Sketch
 *
 * Entry point.  Constructs all subsystem objects and delegates every loop
 * iteration to CommunicationManager and RobotController.
 *
 * Communication transport: Arduino UNO Q RouterBridge/RPClite.
 * This is the sole active communication path.
 *
 * Module dependency graph (arrows = "uses"):
 *
 *   arduino.ino
 *     ├─► CommunicationManager  (Bridge RPC command state)
 *     └─► RobotController       (per-loop orchestrator, all motor actions)
 *           ├─► MotorController
 *           ├─► UltrasonicSensor
 *           ├─► NavigationManager
 *           └─► CommunicationManager
 *
 * setup() sequence:
 *   1. Bridge.begin()   – start UNO Q MPU↔MCU RouterBridge IPC
 *   2. motor.begin()    – configure motor pins, enter stopped state
 *   3. comm.begin()     – initialise command/safety state
 *
 * No alert, LED, or buzzer objects exist in this sketch.
 * No Wi-Fi.  No network credentials.
 */

#include "Arduino_RouterBridge.h"
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

CommunicationManager comm;                       // Handles Bridge RPC command state
RobotController      robot(&motor, &ultrasonic, &comm); // Orchestrates all motion logic

String asm_ping() {
    return comm.rpcPing();
}

String asm_led(String value) {
    return comm.rpcLed(value);
}

String asm_buzzer(String state) {
    return comm.rpcBuzzer(state);
}

String asm_motor(String direction, int speed) {
    return comm.rpcMotor(direction, speed);
}

String asm_mode(String value) {
    return comm.rpcMode(value);
}

String asm_safe_reset() {
    return comm.rpcSafeReset();
}

String asm_control_tick() {
    return comm.rpcControlTick();
}

String asm_manual_hold() {
    return comm.rpcManualHold();
}

String asm_status() {
    return comm.rpcStatus(
        motor.getMovement(),
        motor.getSpeed(),
        ultrasonic.distanceAvailable(0),
        ultrasonic.getFrontDistance(),
        ultrasonic.distanceAvailable(2),
        ultrasonic.getBackDistance(),
        ultrasonic.distanceAvailable(3),
        ultrasonic.getLeftDistance(),
        ultrasonic.distanceAvailable(1),
        ultrasonic.getRightDistance()
    );
}

String asm_ultrasonic_diag() {
    if (comm.getMode() != MODE_MANUAL) {
        return "{\"type\":\"error\",\"error\":\"DIAG_REQUIRES_MANUAL\"}";
    }
    ultrasonic.measureAllOnce();
    return comm.rpcUltrasonicDiagnostic(
        ultrasonic.distanceAvailable(0),
        ultrasonic.getFrontDistance(),
        ultrasonic.distanceAvailable(2),
        ultrasonic.getBackDistance(),
        ultrasonic.distanceAvailable(3),
        ultrasonic.getLeftDistance(),
        ultrasonic.distanceAvailable(1),
        ultrasonic.getRightDistance()
    );
}

String asm_buzzer_tone_diag(int frequency_hz, int duration_ms) {
    if (comm.getMode() != MODE_MANUAL) {
        return "{\"type\":\"error\",\"error\":\"DIAG_REQUIRES_MANUAL\"}";
    }
    return comm.rpcBuzzerToneDiag(frequency_hz, duration_ms);
}

void setup() {
    Bridge.begin();
    Bridge.provide_safe("asm_ping", asm_ping);
    Bridge.provide_safe("asm_led", asm_led);
    Bridge.provide_safe("asm_buzzer", asm_buzzer);
    Bridge.provide_safe("asm_motor", asm_motor);
    Bridge.provide_safe("asm_mode", asm_mode);
    Bridge.provide_safe("asm_safe_reset", asm_safe_reset);
    Bridge.provide_safe("asm_control_tick", asm_control_tick);
    Bridge.provide_safe("asm_manual_hold", asm_manual_hold);
    Bridge.provide_safe("asm_status", asm_status);
    Bridge.provide_safe("asm_ultrasonic_diag", asm_ultrasonic_diag);
    Bridge.provide_safe("asm_buzzer_tone_diag", asm_buzzer_tone_diag);

    motor.begin();                  // Configure motor GPIO pins and enter stopped state
    ultrasonic.begin();             // Configure ultrasonic sensor GPIO pins
    comm.begin();                   // Initialise command/safety state
}

void loop() {
    comm.update();          // Update command timeout, motion lease, and warning state
    robot.update();         // Execute one control cycle (obstacle check, motor commands)
    delay(LOOP_DELAY_MS);   // Fixed loop rate defined in config.h
}
