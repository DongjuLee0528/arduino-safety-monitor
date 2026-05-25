#include "pins.h"
#include "motor.h"
#include "ultrasonic.h"
#include "alert.h"
#include "bridge_rpc.h"

MotorController motor(MOTOR_LF_PIN, MOTOR_LB_PIN, MOTOR_RF_PIN, MOTOR_RB_PIN, MOTOR_LPWM_PIN, MOTOR_RPWM_PIN);
UltrasonicSensor ultrasonic(ULTRASONIC_FRONT_TRIG_PIN, ULTRASONIC_FRONT_ECHO_PIN, ULTRASONIC_BACK_TRIG_PIN, ULTRASONIC_BACK_ECHO_PIN,
                           ULTRASONIC_LEFT_TRIG_PIN, ULTRASONIC_LEFT_ECHO_PIN, ULTRASONIC_RIGHT_TRIG_PIN, ULTRASONIC_RIGHT_ECHO_PIN);
AlertController alert(RED_LED_PIN, BUZZER_PIN);
BridgeRPC bridge(&motor, &alert);

String lastCommand = "";
String currentMotorStatus = "stopped";

void setup() {
  Serial.begin(115200);
  delay(1000);
}

void loop() {
  bridge.update();

  if (bridge.isInSafeMode()) {
    motor.stop();
    alert.startAlert();
    currentMotorStatus = "emergency_stop";
    bridge.sendMotorStatus(currentMotorStatus);
    return;
  }

  ultrasonic.update();

  float frontDistance = ultrasonic.getFrontDistance();
  float rightDistance = ultrasonic.getRightDistance();
  float backDistance = ultrasonic.getBackDistance();
  float leftDistance = ultrasonic.getLeftDistance();

  bridge.sendUltrasonicData(frontDistance, rightDistance, backDistance, leftDistance);

  bool hasObstacle = ultrasonic.hasObstacle(0) || ultrasonic.hasObstacle(1) ||
                     ultrasonic.hasObstacle(2) || ultrasonic.hasObstacle(3);

  if (hasObstacle) {
    String avoidDirection = ultrasonic.getAvoidanceDirection();

    if (avoidDirection == "stop") {
      motor.stop();
      alert.startAlert();
      currentMotorStatus = "obstacle_stop";
    } else {
      motor.processCommand(avoidDirection, 150);
      alert.stopAlert();
      currentMotorStatus = "avoiding_" + avoidDirection;
    }

    bridge.sendAvoidanceDirection(avoidDirection);
  } else {
    alert.stopAlert();
    if (currentMotorStatus.indexOf("avoiding") != -1) {
      motor.stop();
      currentMotorStatus = "stopped";
    }
  }

  bridge.sendMotorStatus(currentMotorStatus);
  alert.update();

  delay(10);
}
