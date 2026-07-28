/*
 * pins.h – Hardware Pin Assignments for Arduino UNO R4 WiFi Robot
 *
 * Defines all GPIO pin numbers for:
 *   - L298N Motor Driver (direction + PWM)
 *   - HC-SR04 Ultrasonic sensors ×4
 *
 * Pin uniqueness table (all digital-mode pins):
 *   Pin 2  – MOTOR_LF
 *   Pin 3  – MOTOR_LB
 *   Pin 4  – MOTOR_RF
 *   Pin 5  – MOTOR_RB
 *   Pin 6  – ULTRASONIC_FRONT_TRIG
 *   Pin 7  – ULTRASONIC_FRONT_ECHO
 *   Pin 8  – ULTRASONIC_BACK_TRIG
 *   Pin 9  – MOTOR_LPWM  (PWM-capable)
 *   Pin 10 – MOTOR_RPWM  (PWM-capable)
 *   Pin 11 – ULTRASONIC_BACK_ECHO
 *   Pin 12 – ULTRASONIC_LEFT_TRIG
 *   Pin 13 – ULTRASONIC_LEFT_ECHO
 *   A0     – ULTRASONIC_RIGHT_TRIG  (used as digital)
 *   A1     – ULTRASONIC_RIGHT_ECHO  (used as digital)
 *
 * No conflicts exist after alert/LED/buzzer pins were removed.
 */

#ifndef PINS_H
#define PINS_H

// L298N Motor Driver
#define MOTOR_LF_PIN    2    // Left  motor forward  direction
#define MOTOR_LB_PIN    3    // Left  motor backward direction
#define MOTOR_RF_PIN    4    // Right motor forward  direction
#define MOTOR_RB_PIN    5    // Right motor backward direction
#define MOTOR_LPWM_PIN  9    // Left  motor PWM speed (must be PWM-capable)
#define MOTOR_RPWM_PIN  10   // Right motor PWM speed (must be PWM-capable)

// HC-SR04 Ultrasonic Sensors
#define ULTRASONIC_FRONT_TRIG_PIN  6   // Front sensor trigger
#define ULTRASONIC_FRONT_ECHO_PIN  7   // Front sensor echo
#define ULTRASONIC_BACK_TRIG_PIN   8   // Rear  sensor trigger
#define ULTRASONIC_BACK_ECHO_PIN   11  // Rear  sensor echo
#define ULTRASONIC_LEFT_TRIG_PIN   12  // Left  sensor trigger
#define ULTRASONIC_LEFT_ECHO_PIN   13  // Left  sensor echo
#define ULTRASONIC_RIGHT_TRIG_PIN  A0  // Right sensor trigger (analog pin as digital)
#define ULTRASONIC_RIGHT_ECHO_PIN  A1  // Right sensor echo   (analog pin as digital)

#endif
