/*
 * config.h – Robot Configuration Constants
 *
 * Single source of truth for all tuneable values.
 * No magic numbers appear anywhere else in the codebase.
 *
 * Sections:
 *   1. Motor control
 *   2. Timing
 *   3. Ultrasonic sensors
 *   4. Navigation / obstacle avoidance
 *   5. Communication
 */

#ifndef CONFIG_H
#define CONFIG_H

// ---------------------------------------------------------------------------
// 1. Motor control
// ---------------------------------------------------------------------------
#define MOTOR_SPEED_MIN        0     // Minimum valid PWM value
#define MOTOR_SPEED_MAX        255   // Maximum valid PWM value
#define MOTOR_SPEED_DEFAULT    150   // Default PWM speed used by setSpeed()
#define CRUISE_SPEED           150   // PWM for autonomous forward cruising
#define OBSTACLE_AVOID_SPEED   150   // PWM for obstacle-avoidance manoeuvres
#define TURN_SPEED             200   // PWM for point turns (left / right)

// ---------------------------------------------------------------------------
// 2. Timing
// ---------------------------------------------------------------------------
#define LOOP_DELAY_MS          10    // Main loop delay in ms (≈100 Hz)

// ---------------------------------------------------------------------------
// 3. Ultrasonic sensors
// ---------------------------------------------------------------------------
#define ULTRASONIC_TIMEOUT_US  30000 // pulseIn() timeout in µs (≈5 m max range)
#define ULTRASONIC_SAMPLES     3     // Rolling-average window size
#define ULTRASONIC_INTERVAL_MS 50    // Minimum ms between consecutive readings
#define MAX_SENSOR_RANGE_CM    300.0 // Value returned when no valid reading exists

// ---------------------------------------------------------------------------
// 4. Navigation / obstacle avoidance
// ---------------------------------------------------------------------------
#define OBSTACLE_THRESHOLD_CM  30.0  // Distance (cm) at which a sensor fires "blocked"

// ---------------------------------------------------------------------------
// 5. Communication – USB Serial / BridgeRPC JSON protocol
// ---------------------------------------------------------------------------
#define SERIAL_BAUD_RATE       115200  // USB Serial baud rate
#define COMM_MAX_LINE_LEN      128     // Maximum accepted input line length in bytes
#define COMM_SEND_INTERVAL_MS  100     // Interval between outgoing telemetry messages (ms)

// If no byte is received from the host within this interval (ms),
// CommunicationManager declares the connection lost (isConnected() → false).
// RobotController then stops motors and forces MANUAL mode.
// The robot resumes only after a new valid command is received.
#define SERIAL_CMD_TIMEOUT_MS  2000

#endif
