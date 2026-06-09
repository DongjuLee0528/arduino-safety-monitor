/*
 * Arduino Configuration Constants
 *
 * This header file defines all configuration constants for the Arduino robot system.
 * Centralizes hardcoded values to make them easily configurable without modifying
 * multiple source files.
 *
 * Categories:
 * - Motor control settings
 * - Timing configuration
 * - Sensor parameters
 * - Alert system settings
 */

#ifndef CONFIG_H
#define CONFIG_H

// Motor control configuration
#define OBSTACLE_AVOID_SPEED 150     // PWM speed for obstacle avoidance movements

// Timing configuration
#define LOOP_DELAY_MS 10             // Main loop delay in milliseconds (100Hz)

// Ultrasonic sensor configuration
#define ULTRASONIC_TIMEOUT_US 30000  // Timeout for pulseIn() in microseconds (~5m range)

// Alert system configuration
#define LED_BLINK_INTERVAL_MS 500    // LED blink interval in milliseconds (1Hz)

#endif