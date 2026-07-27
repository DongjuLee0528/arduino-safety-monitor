/*
 * Arduino UNO R4 WiFi – Main Sketch
 *
 * Entry point.  Constructs all subsystem objects and delegates every loop
 * iteration to RobotController, which coordinates the remaining modules.
 *
 * Module dependency graph (arrows = "uses"):
 *
 *   main.ino
 *     └─► RobotController
 *           ├─► BridgeRPC        (communication / mode / safe-mode)
 *           │     ├─► MotorController  (direct access for safe-stop)
 *           │     └─► AlertController (direct access for safe-stop)
 *           ├─► MotorController  (movement execution)
 *           ├─► UltrasonicSensor (distance readings)
 *           ├─► AlertController  (LED / buzzer)
 *           └─► NavigationManager (state-machine decisions)
 */

#include "config.h"
#include "pins.h"
#include "motor.h"
#include "ultrasonic.h"
#include "alert.h"
#include "bridge_rpc.h"
#include "navigation.h"
#include "robot_controller.h"

MotorController  motor(MOTOR_LF_PIN, MOTOR_LB_PIN,
                       MOTOR_RF_PIN, MOTOR_RB_PIN,
                       MOTOR_LPWM_PIN, MOTOR_RPWM_PIN);

UltrasonicSensor ultrasonic(ULTRASONIC_FRONT_TRIG_PIN, ULTRASONIC_FRONT_ECHO_PIN,
                            ULTRASONIC_BACK_TRIG_PIN,  ULTRASONIC_BACK_ECHO_PIN,
                            ULTRASONIC_LEFT_TRIG_PIN,  ULTRASONIC_LEFT_ECHO_PIN,
                            ULTRASONIC_RIGHT_TRIG_PIN, ULTRASONIC_RIGHT_ECHO_PIN);

AlertController  alert(RED_LED_PIN, BUZZER_PIN);
BridgeRPC        bridge(&motor, &alert);
RobotController  robot(&motor, &ultrasonic, &alert, &bridge);

void setup() {
    Serial.begin(115200);
    delay(1000);
}

void loop() {
    bridge.update();
    robot.update();
    alert.update();
    delay(LOOP_DELAY_MS);
}
