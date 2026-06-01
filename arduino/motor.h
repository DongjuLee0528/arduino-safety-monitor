/*
 * Motor Controller Library for L298N Motor Driver
 *
 * This library provides high-level control functions for a differential drive robot
 * using the L298N dual H-bridge motor driver. It supports:
 * - Bidirectional movement (forward/backward)
 * - Turning in place (left/right)
 * - Variable speed control via PWM
 * - Command string processing for remote control
 *
 * Hardware setup:
 * - Two DC motors connected to L298N driver
 * - Direction control pins for each motor
 * - PWM pins for speed control
 */

#ifndef MOTOR_H
#define MOTOR_H

/**
 * MotorController Class
 *
 * Manages L298N motor driver for differential drive robot control.
 * Provides methods for basic movement and speed control.
 */
class MotorController {
private:
    // Pin assignments for motor direction control
    int leftForward;   // Left motor forward direction pin
    int leftBackward;  // Left motor backward direction pin
    int rightForward;  // Right motor forward direction pin
    int rightBackward; // Right motor backward direction pin

    // PWM pin assignments for speed control
    int leftPWM;       // Left motor PWM speed control pin
    int rightPWM;      // Right motor PWM speed control pin

public:
    /**
     * Constructor - Initialize motor controller with pin assignments
     * @param lf: Left motor forward pin
     * @param lb: Left motor backward pin
     * @param rf: Right motor forward pin
     * @param rb: Right motor backward pin
     * @param lpwm: Left motor PWM pin
     * @param rpwm: Right motor PWM pin
     */
    MotorController(int lf, int lb, int rf, int rb, int lpwm, int rpwm) {
        leftForward = lf;
        leftBackward = lb;
        rightForward = rf;
        rightBackward = rb;
        leftPWM = lpwm;
        rightPWM = rpwm;

        // Configure all pins as outputs
        pinMode(leftForward, OUTPUT);
        pinMode(leftBackward, OUTPUT);
        pinMode(rightForward, OUTPUT);
        pinMode(rightBackward, OUTPUT);
        pinMode(leftPWM, OUTPUT);
        pinMode(rightPWM, OUTPUT);
    }

    /**
     * Move robot forward at specified speed
     * @param speed: PWM value (0-255), default 255 (full speed)
     */
    void forward(int speed = 255) {
        digitalWrite(leftForward, HIGH);   // Left motor forward
        digitalWrite(leftBackward, LOW);
        digitalWrite(rightForward, HIGH);  // Right motor forward
        digitalWrite(rightBackward, LOW);
        analogWrite(leftPWM, speed);       // Set speed for both motors
        analogWrite(rightPWM, speed);
    }

    /**
     * Move robot backward at specified speed
     * @param speed: PWM value (0-255), default 255 (full speed)
     */
    void backward(int speed = 255) {
        digitalWrite(leftForward, LOW);    // Left motor backward
        digitalWrite(leftBackward, HIGH);
        digitalWrite(rightForward, LOW);   // Right motor backward
        digitalWrite(rightBackward, HIGH);
        analogWrite(leftPWM, speed);       // Set speed for both motors
        analogWrite(rightPWM, speed);
    }

    /**
     * Turn robot left (counterclockwise) by rotating motors in opposite directions
     * @param speed: PWM value (0-255), default 200 (reduced speed for control)
     */
    void turnLeft(int speed = 200) {
        digitalWrite(leftForward, LOW);    // Left motor backward
        digitalWrite(leftBackward, HIGH);
        digitalWrite(rightForward, HIGH);  // Right motor forward
        digitalWrite(rightBackward, LOW);
        analogWrite(leftPWM, speed);       // Set speed for both motors
        analogWrite(rightPWM, speed);
    }

    /**
     * Turn robot right (clockwise) by rotating motors in opposite directions
     * @param speed: PWM value (0-255), default 200 (reduced speed for control)
     */
    void turnRight(int speed = 200) {
        digitalWrite(leftForward, HIGH);   // Left motor forward
        digitalWrite(leftBackward, LOW);
        digitalWrite(rightForward, LOW);   // Right motor backward
        digitalWrite(rightBackward, HIGH);
        analogWrite(leftPWM, speed);       // Set speed for both motors
        analogWrite(rightPWM, speed);
    }

    /**
     * Stop robot by setting both direction pins HIGH (brake mode)
     * This creates a braking effect rather than just coasting to a stop
     */
    void stop() {
        digitalWrite(leftForward, HIGH);   // Brake mode - both pins HIGH
        digitalWrite(leftBackward, HIGH);
        digitalWrite(rightForward, HIGH);  // Brake mode - both pins HIGH
        digitalWrite(rightBackward, HIGH);
        analogWrite(leftPWM, 0);          // Zero PWM to ensure stop
        analogWrite(rightPWM, 0);
    }

    /**
     * Process movement command from string input
     * Supports both full words and single character commands
     * @param command: Movement command string
     * @param speed: PWM value (0-255), default 200
     */
    void processCommand(String command, int speed = 200) {
        if (command == "forward" || command == "F") {
            forward(speed);
        } else if (command == "backward" || command == "B") {
            backward(speed);
        } else if (command == "left" || command == "L") {
            turnLeft(speed);
        } else if (command == "right" || command == "R") {
            turnRight(speed);
        } else if (command == "stop" || command == "S") {
            stop();
        }
        // Invalid commands are ignored
    }
};

#endif
