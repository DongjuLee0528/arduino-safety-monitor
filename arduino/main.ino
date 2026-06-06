/*
 * Arduino UNO Q Main Controller
 *
 * This is the main program file for an autonomous robot system that integrates:
 * - Motor control for movement
 * - Ultrasonic sensors for obstacle detection
 * - Alert system with LED and buzzer
 * - Bridge communication for remote control and monitoring
 *
 * The robot operates in two modes:
 * 1. Autonomous obstacle avoidance mode
 * 2. Safe mode (emergency stop triggered remotely)
 */

#include "pins.h"       // Hardware pin definitions
#include "motor.h"      // Motor control functionality
#include "ultrasonic.h" // Ultrasonic sensor management
#include "alert.h"      // LED and buzzer alert system
#include "bridge_rpc.h" // Communication bridge with external systems

// Initialize hardware controllers with pin assignments
MotorController motor(MOTOR_LF_PIN, MOTOR_LB_PIN, MOTOR_RF_PIN, MOTOR_RB_PIN, MOTOR_LPWM_PIN, MOTOR_RPWM_PIN);
UltrasonicSensor ultrasonic(ULTRASONIC_FRONT_TRIG_PIN, ULTRASONIC_FRONT_ECHO_PIN, ULTRASONIC_BACK_TRIG_PIN, ULTRASONIC_BACK_ECHO_PIN,
                           ULTRASONIC_LEFT_TRIG_PIN, ULTRASONIC_LEFT_ECHO_PIN, ULTRASONIC_RIGHT_TRIG_PIN, ULTRASONIC_RIGHT_ECHO_PIN);
AlertController alert(RED_LED_PIN, BUZZER_PIN);
BridgeRPC bridge(&motor, &alert);

// Global state variables
String lastCommand = "";              // Store the last received command for debugging
String currentMotorStatus = "stopped"; // Track current motor state for status reporting

/**
 * Arduino setup function - runs once at startup
 * Initializes serial communication and waits for system stabilization
 */
void setup() {
  Serial.begin(115200);  // Initialize serial communication at 115200 baud
  delay(1000);           // Allow system to stabilize before starting main loop
}

/**
 * Main program loop - runs continuously after setup()
 * Handles obstacle detection, avoidance, and communication with external systems
 */
void loop() {
  // Process incoming bridge communications
  bridge.update();

  // Check if system is in emergency safe mode (remotely triggered)
  if (bridge.isInSafeMode()) {
    motor.stop();                        // Stop all motor movement
    alert.startAlert();                  // Activate warning alerts
    currentMotorStatus = "emergency_stop"; // Update status
    bridge.sendMotorStatus(currentMotorStatus); // Notify external systems
    return;                              // Skip the rest of the loop
  }

  // Process remote motor commands if received
  if (bridge.hasMotorCommand()) {
    String remoteDirection = bridge.getMotorDirection();
    int remoteSpeed = bridge.getMotorSpeed();

    motor.processCommand(remoteDirection, remoteSpeed);
    currentMotorStatus = "remote_" + remoteDirection;
    bridge.sendMotorStatus(currentMotorStatus);

    // Skip autonomous behavior when under remote control
    return;
  }

  // Update all ultrasonic sensor readings
  ultrasonic.update();

  // Get distance measurements from all four sensors
  float frontDistance = ultrasonic.getFrontDistance();
  float rightDistance = ultrasonic.getRightDistance();
  float backDistance = ultrasonic.getBackDistance();
  float leftDistance = ultrasonic.getLeftDistance();

  // Send sensor data to external monitoring systems
  bridge.sendUltrasonicData(frontDistance, rightDistance, backDistance, leftDistance);

  // Check if any sensor detects an obstacle (indices: 0=front, 1=right, 2=back, 3=left)
  bool hasObstacle = ultrasonic.hasObstacle(0) || ultrasonic.hasObstacle(1) ||
                     ultrasonic.hasObstacle(2) || ultrasonic.hasObstacle(3);

  // Handle obstacle detection and avoidance
  if (hasObstacle) {
    // Get recommended avoidance direction from ultrasonic sensor logic
    String avoidDirection = ultrasonic.getAvoidanceDirection();

    if (avoidDirection == "stop") {
      // No safe direction found - stop and alert
      motor.stop();
      alert.startAlert();
      currentMotorStatus = "obstacle_stop";
    } else {
      // Safe direction found - move in that direction
      motor.processCommand(avoidDirection, 150); // Move with PWM speed 150
      alert.stopAlert();                         // Stop any active alerts
      currentMotorStatus = "avoiding_" + avoidDirection;
    }

    // Inform external systems about avoidance action
    bridge.sendAvoidanceDirection(avoidDirection);
  } else {
    // No obstacles detected
    alert.stopAlert(); // Ensure alerts are off

    // If we were previously in avoidance mode, stop the motors
    if (currentMotorStatus.indexOf("avoiding") != -1) {
      motor.stop();
      currentMotorStatus = "stopped";
    }
  }

  // Send current motor status to external monitoring systems
  bridge.sendMotorStatus(currentMotorStatus);

  // Update alert system (handles LED and buzzer states)
  alert.update();

  // Small delay to prevent overwhelming the system (100Hz loop rate)
  delay(10);
}
