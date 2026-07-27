/*
 * Robot Configuration Constants
 *
 * Single source of truth for all tuneable values.
 * No magic numbers should appear anywhere else in the codebase.
 *
 * Sections:
 *   1. Motor control
 *   2. Timing
 *   3. Ultrasonic sensors
 *   4. Navigation / obstacle avoidance
 *   5. Alert system
 */

#ifndef CONFIG_H
#define CONFIG_H

// ---------------------------------------------------------------------------
// 1. Motor control
// ---------------------------------------------------------------------------
#define CRUISE_SPEED           150   // PWM (0-255) for autonomous forward cruising
#define OBSTACLE_AVOID_SPEED   150   // PWM (0-255) for obstacle-avoidance manoeuvres
#define TURN_SPEED             200   // PWM (0-255) for point turns (left / right)
#define MANUAL_DEFAULT_SPEED   200   // PWM (0-255) used when no speed is specified

// ---------------------------------------------------------------------------
// 2. Timing
// ---------------------------------------------------------------------------
#define LOOP_DELAY_MS          10    // Main loop delay in ms (≈100 Hz)

// ---------------------------------------------------------------------------
// 3. Ultrasonic sensors
// ---------------------------------------------------------------------------
#define ULTRASONIC_TIMEOUT_US  30000 // pulseIn() timeout in µs (≈5 m max range)
#define ULTRASONIC_SAMPLES     3     // Number of readings averaged per sensor
#define ULTRASONIC_INTERVAL_MS 50    // Min. milliseconds between consecutive readings

// ---------------------------------------------------------------------------
// 4. Navigation / obstacle avoidance
// ---------------------------------------------------------------------------
#define OBSTACLE_THRESHOLD_CM  30.0  // Distance (cm) at which a sensor fires "blocked"
#define MAX_SENSOR_RANGE_CM    300.0 // Distance returned when no echo is received

// ---------------------------------------------------------------------------
// 5. Alert system
// ---------------------------------------------------------------------------
#define LED_BLINK_INTERVAL_MS  500   // LED blink period in ms (1 Hz, 50 % duty)

#endif
