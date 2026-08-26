/*
 * config.h – Robot Configuration Constants
 *
 * Single source of truth for firmware tuneable values.
 * Hardware calibration and safety timing constants belong here.
 *
 * Sections:
 *   1. Motor control
 *   2. Timing
 *   3. Ultrasonic sensors
 *   4. Navigation / obstacle avoidance
 *   5. Buzzer diagnostic
 *   6. Communication
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
#define LEFT_MOTOR_REVERSED    true  // Invert left channel polarity for mirrored motor mounting
#define HELMET_WARNING_BUZZER_DURATION_MS 5000  // Helmet warning buzzer/blink duration
#define LED_BLINK_INTERVAL_MS  250   // Helmet warning LED toggle interval

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
#define OBSTACLE_THRESHOLD_CM  25.0  // Distance (cm) at which a sensor fires "blocked"
#define SENSOR_STALE_TIMEOUT_MS 1000 // Max age (ms) of a valid reading; older readings are treated as unavailable

// Motion lease – separate from the serial connection keepalive.
// Only the Python main control loop may renew this; generic ping cannot.
// If no control_tick arrives within this window, CommunicationManager
// latches a durable STOP before processing any further commands.
#define MOTION_LEASE_TIMEOUT_MS 1000
#define MANUAL_HOLD_TIMEOUT_MS  500  // Max ms a manual movement may continue without a manual_hold refresh

// ---------------------------------------------------------------------------
// 5. Buzzer
// ---------------------------------------------------------------------------
#define BUZZER_WARN_FREQ_HZ            2000  // Production warning tone frequency (Hz)
#define BUZZER_WARN_HALF_PERIOD_MS     500   // Warning tone on/off half-period (ms)
#define BUZZER_TEST_DURATION_MS        250   // BZR test beep duration (ms)
#define BUZZER_DIAG_DURATION_MIN_MS    100   // Minimum buzzer diagnostic pulse duration (ms)
#define BUZZER_DIAG_DURATION_MAX_MS    1000  // Maximum buzzer diagnostic pulse duration (ms)
#define BUZZER_TONE_DIAG_FREQUENCY_MIN_HZ  500   // Minimum buzzer tone diagnostic frequency (Hz)
#define BUZZER_TONE_DIAG_FREQUENCY_MAX_HZ  5000  // Maximum buzzer tone diagnostic frequency (Hz)

// ---------------------------------------------------------------------------
// 6. Communication – UNO Q RouterBridge / BridgeRPC command protocol
// ---------------------------------------------------------------------------
#define COMM_SEND_INTERVAL_MS  100     // Interval between outgoing telemetry messages (ms)

// If no valid Bridge command is received from the host within this interval
// (ms), CommunicationManager declares the connection lost (isConnected() →
// false).
// RobotController then stops motors and forces MANUAL mode.
// The robot resumes only after a new valid command is received.
#define COMMAND_TIMEOUT_MS     2000

#endif
