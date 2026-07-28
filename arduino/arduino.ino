/*
 * arduino.ino – Arduino UNO R4 WiFi Main Sketch
 *
 * Entry point.  Constructs all subsystem objects and delegates every loop
 * iteration to CommunicationManager and RobotController.
 *
 * Communication transport: Wi-Fi TCP (WiFiS3).
 * USB Serial is used for debug diagnostics only.
 *
 * Module dependency graph (arrows = "uses"):
 *
 *   arduino.ino
 *     ├─► CommunicationManager  (Wi-Fi transport, command parsing, state)
 *     └─► RobotController       (per-loop orchestrator, all motor actions)
 *           ├─► MotorController
 *           ├─► UltrasonicSensor
 *           ├─► NavigationManager
 *           └─► CommunicationManager
 *
 * setup() sequence:
 *   1. Serial.begin()   – debug port
 *   2. motor.begin()    – configure motor pins, enter stopped state
 *   3. WiFi.begin()     – associate with network (blocks up to WIFI_CONNECT_TIMEOUT_MS)
 *   4. comm.begin()     – start TCP server
 *
 * No alert, LED, or buzzer objects exist in this sketch.
 */

#include <WiFiS3.h>
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

    Serial.print("WIFI:CONNECTING ssid=");
    Serial.println(WIFI_SSID);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - wifiStart > WIFI_CONNECT_TIMEOUT_MS) {
            Serial.println("WIFI:CONNECT_TIMEOUT");
            break;
        }
        delay(500);
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("WIFI:CONNECTED ip=");
        Serial.println(WiFi.localIP());
        comm.begin();
    } else {
        Serial.println("WIFI:FAILED motors remain stopped");
    }
}

void loop() {
    comm.update();
    robot.update();
    delay(LOOP_DELAY_MS);
}
