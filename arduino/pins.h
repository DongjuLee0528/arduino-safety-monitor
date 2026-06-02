/*
 * Pin Definitions for Arduino UNO Q Robot
 *
 * This header file defines all the hardware pin assignments for:
 * - L298N Motor Driver connections
 * - HC-SR04 Ultrasonic sensors (4 directional sensors)
 * - Alert system (LED and buzzer)
 *
 * Pin Layout:
 * Digital pins 2-13: Motors and most sensors
 * Analog pins A0-A2: Right ultrasonic sensor and buzzer
 * Built-in LED: Status indicator
 */

#ifndef PINS_H
#define PINS_H

// L298N Motor Driver Pin Assignments
// Controls left and right motors with direction and PWM speed control
#define MOTOR_LF_PIN 2     // Left motor forward direction pin
#define MOTOR_LB_PIN 3     // Left motor backward direction pin
#define MOTOR_RF_PIN 4     // Right motor forward direction pin
#define MOTOR_RB_PIN 5     // Right motor backward direction pin
#define MOTOR_LPWM_PIN 9   // Left motor PWM speed control pin
#define MOTOR_RPWM_PIN 10  // Right motor PWM speed control pin

// HC-SR04 Ultrasonic Sensor Pin Assignments
// Four sensors positioned for 360-degree obstacle detection
#define ULTRASONIC_FRONT_TRIG_PIN 6   // Front sensor trigger pin
#define ULTRASONIC_FRONT_ECHO_PIN 7   // Front sensor echo pin
#define ULTRASONIC_BACK_TRIG_PIN 8    // Back sensor trigger pin
#define ULTRASONIC_BACK_ECHO_PIN 11   // Back sensor echo pin
#define ULTRASONIC_LEFT_TRIG_PIN 12   // Left sensor trigger pin
#define ULTRASONIC_LEFT_ECHO_PIN 13   // Left sensor echo pin
#define ULTRASONIC_RIGHT_TRIG_PIN A0  // Right sensor trigger pin (analog pin used as digital)
#define ULTRASONIC_RIGHT_ECHO_PIN A1  // Right sensor echo pin (analog pin used as digital)

// Alert System Pin Assignments
// Visual and audio feedback for system status and warnings
#define RED_LED_PIN LED_BUILTIN  // Built-in LED for status indication
#define BUZZER_PIN A2            // Buzzer for audio alerts

#endif