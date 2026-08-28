/*
 * pins.h – Hardware Pin Assignments for Arduino UNO Q Robot
 *
 * Defines all GPIO pin numbers for:
 *   - L298N Motor Driver (direction + PWM)
 *   - HC-SR04 Ultrasonic sensors ×4
 *
 * L298N wiring notes:
 *   - Remove ENA and ENB jumpers; PWM speed control is applied via ENA/ENB.
 *   - OUT1/OUT2 drive the left-front and left-rear TT motors (left channel).
 *   - OUT3/OUT4 drive the right-front and right-rear TT motors (right channel).
 *   - All grounds (Arduino, L298N, battery) must share a common ground.
 *   - HC-SR04 sensors are powered from the 5 V bus.
 *   - A0–A3 are used as digital GPIO (no ADC reads on these pins).
 *
 * Reserved pins (do not assign):
 *   D0  – UART RX
 *   D1  – UART TX
 *   D3  – ALERT_LED_PIN
 *   A4  – I2C SDA
 *   A5  – I2C SCL
 *
 * Active pin uniqueness table:
 *   D2  – MOTOR_LEFT_IN1_PIN
 *   D4  – MOTOR_LEFT_IN2_PIN
 *   D5  – MOTOR_LEFT_ENABLE_PIN   (PWM)
 *   D6  – MOTOR_RIGHT_ENABLE_PIN  (PWM)
 *   D7  – MOTOR_RIGHT_IN1_PIN
 *   D8  – MOTOR_RIGHT_IN2_PIN
 *   D9  – ULTRASONIC_FRONT_TRIGGER_PIN
 *   D10 – ULTRASONIC_FRONT_ECHO_PIN
 *   D11 – ULTRASONIC_REAR_TRIGGER_PIN
 *   D12 – ULTRASONIC_REAR_ECHO_PIN
 *   D13 – BUZZER_DIAG_PIN         (diagnostic/test tone only)
 *   A0  – ULTRASONIC_LEFT_TRIGGER_PIN  (digital GPIO)
 *   A1  – ULTRASONIC_LEFT_ECHO_PIN     (digital GPIO)
 *   A2  – ULTRASONIC_RIGHT_TRIGGER_PIN (digital GPIO)
 *   A3  – ULTRASONIC_RIGHT_ECHO_PIN    (digital GPIO)
 */

#ifndef PINS_H
#define PINS_H

// ---------------------------------------------------------------------------
// L298N Motor Driver
// ---------------------------------------------------------------------------
#define MOTOR_LEFT_ENABLE_PIN   5    // ENA – left  channel PWM speed (PWM-capable)
#define MOTOR_LEFT_IN1_PIN      2    // IN1 – left  channel direction A
#define MOTOR_LEFT_IN2_PIN      4    // IN2 – left  channel direction B
#define MOTOR_RIGHT_ENABLE_PIN  6    // ENB – right channel PWM speed (PWM-capable)
#define MOTOR_RIGHT_IN1_PIN     7    // IN3 – right channel direction A
#define MOTOR_RIGHT_IN2_PIN     8    // IN4 – right channel direction B
#define ALERT_LED_PIN           3    // D3 – red helmet-warning LED control output (PWM-capable)
#define BUZZER_DIAG_PIN         13   // D13 – buzzer diagnostic signal output (active-HIGH, temporary diagnostic use)

// ---------------------------------------------------------------------------
// HC-SR04 Ultrasonic Sensors
// A0–A3 are used as digital GPIO; no ADC reads occur on these pins.
// ---------------------------------------------------------------------------
#define ULTRASONIC_FRONT_TRIGGER_PIN   9    // Front sensor trigger
#define ULTRASONIC_FRONT_ECHO_PIN      10   // Front sensor echo
#define ULTRASONIC_REAR_TRIGGER_PIN    11   // Rear  sensor trigger
#define ULTRASONIC_REAR_ECHO_PIN       12   // Rear  sensor echo
#define ULTRASONIC_LEFT_TRIGGER_PIN    A0   // Left  sensor trigger (analog pin as digital)
#define ULTRASONIC_LEFT_ECHO_PIN       A1   // Left  sensor echo   (analog pin as digital)
#define ULTRASONIC_RIGHT_TRIGGER_PIN   A2   // Right sensor trigger (analog pin as digital)
#define ULTRASONIC_RIGHT_ECHO_PIN      A3   // Right sensor echo   (analog pin as digital)

// ---------------------------------------------------------------------------
// Compile-time duplicate-pin detection
// Each active pin is assigned a unique integer; the static_assert checks that
// no two constants share the same value.
// ---------------------------------------------------------------------------
static_assert(MOTOR_LEFT_ENABLE_PIN   != MOTOR_LEFT_IN1_PIN,          "Pin conflict: LEFT_ENABLE / LEFT_IN1");
static_assert(MOTOR_LEFT_ENABLE_PIN   != MOTOR_LEFT_IN2_PIN,          "Pin conflict: LEFT_ENABLE / LEFT_IN2");
static_assert(MOTOR_LEFT_ENABLE_PIN   != MOTOR_RIGHT_ENABLE_PIN,      "Pin conflict: LEFT_ENABLE / RIGHT_ENABLE");
static_assert(MOTOR_LEFT_ENABLE_PIN   != MOTOR_RIGHT_IN1_PIN,         "Pin conflict: LEFT_ENABLE / RIGHT_IN1");
static_assert(MOTOR_LEFT_ENABLE_PIN   != MOTOR_RIGHT_IN2_PIN,         "Pin conflict: LEFT_ENABLE / RIGHT_IN2");
static_assert(MOTOR_LEFT_ENABLE_PIN   != ALERT_LED_PIN,               "Pin conflict: LEFT_ENABLE / ALERT_LED");
static_assert(MOTOR_LEFT_IN1_PIN      != MOTOR_LEFT_IN2_PIN,          "Pin conflict: LEFT_IN1 / LEFT_IN2");
static_assert(MOTOR_LEFT_IN1_PIN      != MOTOR_RIGHT_ENABLE_PIN,      "Pin conflict: LEFT_IN1 / RIGHT_ENABLE");
static_assert(MOTOR_LEFT_IN1_PIN      != MOTOR_RIGHT_IN1_PIN,         "Pin conflict: LEFT_IN1 / RIGHT_IN1");
static_assert(MOTOR_LEFT_IN1_PIN      != MOTOR_RIGHT_IN2_PIN,         "Pin conflict: LEFT_IN1 / RIGHT_IN2");
static_assert(MOTOR_LEFT_IN1_PIN      != ALERT_LED_PIN,               "Pin conflict: LEFT_IN1 / ALERT_LED");
static_assert(MOTOR_LEFT_IN2_PIN      != MOTOR_RIGHT_ENABLE_PIN,      "Pin conflict: LEFT_IN2 / RIGHT_ENABLE");
static_assert(MOTOR_LEFT_IN2_PIN      != MOTOR_RIGHT_IN1_PIN,         "Pin conflict: LEFT_IN2 / RIGHT_IN1");
static_assert(MOTOR_LEFT_IN2_PIN      != MOTOR_RIGHT_IN2_PIN,         "Pin conflict: LEFT_IN2 / RIGHT_IN2");
static_assert(MOTOR_LEFT_IN2_PIN      != ALERT_LED_PIN,               "Pin conflict: LEFT_IN2 / ALERT_LED");
static_assert(MOTOR_RIGHT_ENABLE_PIN  != MOTOR_RIGHT_IN1_PIN,         "Pin conflict: RIGHT_ENABLE / RIGHT_IN1");
static_assert(MOTOR_RIGHT_ENABLE_PIN  != MOTOR_RIGHT_IN2_PIN,         "Pin conflict: RIGHT_ENABLE / RIGHT_IN2");
static_assert(MOTOR_RIGHT_ENABLE_PIN  != ALERT_LED_PIN,               "Pin conflict: RIGHT_ENABLE / ALERT_LED");
static_assert(MOTOR_RIGHT_IN1_PIN     != MOTOR_RIGHT_IN2_PIN,         "Pin conflict: RIGHT_IN1 / RIGHT_IN2");
static_assert(MOTOR_RIGHT_IN1_PIN     != ALERT_LED_PIN,               "Pin conflict: RIGHT_IN1 / ALERT_LED");
static_assert(MOTOR_RIGHT_IN2_PIN     != ALERT_LED_PIN,               "Pin conflict: RIGHT_IN2 / ALERT_LED");

static_assert(ULTRASONIC_FRONT_TRIGGER_PIN != ULTRASONIC_FRONT_ECHO_PIN,   "Pin conflict: FRONT_TRIG / FRONT_ECHO");
static_assert(ULTRASONIC_FRONT_TRIGGER_PIN != ALERT_LED_PIN,               "Pin conflict: FRONT_TRIG / ALERT_LED");
static_assert(ULTRASONIC_FRONT_TRIGGER_PIN != ULTRASONIC_REAR_TRIGGER_PIN, "Pin conflict: FRONT_TRIG / REAR_TRIG");
static_assert(ULTRASONIC_FRONT_TRIGGER_PIN != ULTRASONIC_REAR_ECHO_PIN,    "Pin conflict: FRONT_TRIG / REAR_ECHO");
static_assert(ULTRASONIC_FRONT_TRIGGER_PIN != ULTRASONIC_LEFT_TRIGGER_PIN, "Pin conflict: FRONT_TRIG / LEFT_TRIG");
static_assert(ULTRASONIC_FRONT_TRIGGER_PIN != ULTRASONIC_LEFT_ECHO_PIN,    "Pin conflict: FRONT_TRIG / LEFT_ECHO");
static_assert(ULTRASONIC_FRONT_TRIGGER_PIN != ULTRASONIC_RIGHT_TRIGGER_PIN,"Pin conflict: FRONT_TRIG / RIGHT_TRIG");
static_assert(ULTRASONIC_FRONT_TRIGGER_PIN != ULTRASONIC_RIGHT_ECHO_PIN,   "Pin conflict: FRONT_TRIG / RIGHT_ECHO");
static_assert(ULTRASONIC_FRONT_ECHO_PIN    != ULTRASONIC_REAR_TRIGGER_PIN, "Pin conflict: FRONT_ECHO / REAR_TRIG");
static_assert(ULTRASONIC_FRONT_ECHO_PIN    != ALERT_LED_PIN,               "Pin conflict: FRONT_ECHO / ALERT_LED");
static_assert(ULTRASONIC_FRONT_ECHO_PIN    != ULTRASONIC_REAR_ECHO_PIN,    "Pin conflict: FRONT_ECHO / REAR_ECHO");
static_assert(ULTRASONIC_FRONT_ECHO_PIN    != ULTRASONIC_LEFT_TRIGGER_PIN, "Pin conflict: FRONT_ECHO / LEFT_TRIG");
static_assert(ULTRASONIC_FRONT_ECHO_PIN    != ULTRASONIC_LEFT_ECHO_PIN,    "Pin conflict: FRONT_ECHO / LEFT_ECHO");
static_assert(ULTRASONIC_FRONT_ECHO_PIN    != ULTRASONIC_RIGHT_TRIGGER_PIN,"Pin conflict: FRONT_ECHO / RIGHT_TRIG");
static_assert(ULTRASONIC_FRONT_ECHO_PIN    != ULTRASONIC_RIGHT_ECHO_PIN,   "Pin conflict: FRONT_ECHO / RIGHT_ECHO");
static_assert(ULTRASONIC_REAR_TRIGGER_PIN  != ULTRASONIC_REAR_ECHO_PIN,    "Pin conflict: REAR_TRIG / REAR_ECHO");
static_assert(ULTRASONIC_REAR_TRIGGER_PIN  != ALERT_LED_PIN,               "Pin conflict: REAR_TRIG / ALERT_LED");
static_assert(ULTRASONIC_REAR_TRIGGER_PIN  != ULTRASONIC_LEFT_TRIGGER_PIN, "Pin conflict: REAR_TRIG / LEFT_TRIG");
static_assert(ULTRASONIC_REAR_TRIGGER_PIN  != ULTRASONIC_LEFT_ECHO_PIN,    "Pin conflict: REAR_TRIG / LEFT_ECHO");
static_assert(ULTRASONIC_REAR_TRIGGER_PIN  != ULTRASONIC_RIGHT_TRIGGER_PIN,"Pin conflict: REAR_TRIG / RIGHT_TRIG");
static_assert(ULTRASONIC_REAR_TRIGGER_PIN  != ULTRASONIC_RIGHT_ECHO_PIN,   "Pin conflict: REAR_TRIG / RIGHT_ECHO");
static_assert(ULTRASONIC_REAR_ECHO_PIN     != ULTRASONIC_LEFT_TRIGGER_PIN, "Pin conflict: REAR_ECHO / LEFT_TRIG");
static_assert(ULTRASONIC_REAR_ECHO_PIN     != ALERT_LED_PIN,               "Pin conflict: REAR_ECHO / ALERT_LED");
static_assert(ULTRASONIC_REAR_ECHO_PIN     != ULTRASONIC_LEFT_ECHO_PIN,    "Pin conflict: REAR_ECHO / LEFT_ECHO");
static_assert(ULTRASONIC_REAR_ECHO_PIN     != ULTRASONIC_RIGHT_TRIGGER_PIN,"Pin conflict: REAR_ECHO / RIGHT_TRIG");
static_assert(ULTRASONIC_REAR_ECHO_PIN     != ULTRASONIC_RIGHT_ECHO_PIN,   "Pin conflict: REAR_ECHO / RIGHT_ECHO");
static_assert(ULTRASONIC_LEFT_TRIGGER_PIN  != ULTRASONIC_LEFT_ECHO_PIN,    "Pin conflict: LEFT_TRIG / LEFT_ECHO");
static_assert(ULTRASONIC_LEFT_TRIGGER_PIN  != ALERT_LED_PIN,               "Pin conflict: LEFT_TRIG / ALERT_LED");
static_assert(ULTRASONIC_LEFT_TRIGGER_PIN  != ULTRASONIC_RIGHT_TRIGGER_PIN,"Pin conflict: LEFT_TRIG / RIGHT_TRIG");
static_assert(ULTRASONIC_LEFT_TRIGGER_PIN  != ULTRASONIC_RIGHT_ECHO_PIN,   "Pin conflict: LEFT_TRIG / RIGHT_ECHO");
static_assert(ULTRASONIC_LEFT_ECHO_PIN     != ULTRASONIC_RIGHT_TRIGGER_PIN,"Pin conflict: LEFT_ECHO / RIGHT_TRIG");
static_assert(ULTRASONIC_LEFT_ECHO_PIN     != ALERT_LED_PIN,               "Pin conflict: LEFT_ECHO / ALERT_LED");
static_assert(ULTRASONIC_LEFT_ECHO_PIN     != ULTRASONIC_RIGHT_ECHO_PIN,   "Pin conflict: LEFT_ECHO / RIGHT_ECHO");
static_assert(ULTRASONIC_RIGHT_TRIGGER_PIN != ULTRASONIC_RIGHT_ECHO_PIN,   "Pin conflict: RIGHT_TRIG / RIGHT_ECHO");
static_assert(ULTRASONIC_RIGHT_TRIGGER_PIN != ALERT_LED_PIN,               "Pin conflict: RIGHT_TRIG / ALERT_LED");
static_assert(ULTRASONIC_RIGHT_ECHO_PIN    != ALERT_LED_PIN,               "Pin conflict: RIGHT_ECHO / ALERT_LED");

static_assert(BUZZER_DIAG_PIN != MOTOR_LEFT_ENABLE_PIN,                    "Pin conflict: BUZZER_DIAG / LEFT_ENABLE");
static_assert(BUZZER_DIAG_PIN != MOTOR_LEFT_IN1_PIN,                       "Pin conflict: BUZZER_DIAG / LEFT_IN1");
static_assert(BUZZER_DIAG_PIN != MOTOR_LEFT_IN2_PIN,                       "Pin conflict: BUZZER_DIAG / LEFT_IN2");
static_assert(BUZZER_DIAG_PIN != MOTOR_RIGHT_ENABLE_PIN,                   "Pin conflict: BUZZER_DIAG / RIGHT_ENABLE");
static_assert(BUZZER_DIAG_PIN != MOTOR_RIGHT_IN1_PIN,                      "Pin conflict: BUZZER_DIAG / RIGHT_IN1");
static_assert(BUZZER_DIAG_PIN != MOTOR_RIGHT_IN2_PIN,                      "Pin conflict: BUZZER_DIAG / RIGHT_IN2");
static_assert(BUZZER_DIAG_PIN != ALERT_LED_PIN,                            "Pin conflict: BUZZER_DIAG / ALERT_LED");
static_assert(BUZZER_DIAG_PIN != ULTRASONIC_FRONT_TRIGGER_PIN,             "Pin conflict: BUZZER_DIAG / FRONT_TRIG");
static_assert(BUZZER_DIAG_PIN != ULTRASONIC_FRONT_ECHO_PIN,                "Pin conflict: BUZZER_DIAG / FRONT_ECHO");
static_assert(BUZZER_DIAG_PIN != ULTRASONIC_REAR_TRIGGER_PIN,              "Pin conflict: BUZZER_DIAG / REAR_TRIG");
static_assert(BUZZER_DIAG_PIN != ULTRASONIC_REAR_ECHO_PIN,                 "Pin conflict: BUZZER_DIAG / REAR_ECHO");
static_assert(BUZZER_DIAG_PIN != ULTRASONIC_LEFT_TRIGGER_PIN,              "Pin conflict: BUZZER_DIAG / LEFT_TRIG");
static_assert(BUZZER_DIAG_PIN != ULTRASONIC_LEFT_ECHO_PIN,                 "Pin conflict: BUZZER_DIAG / LEFT_ECHO");
static_assert(BUZZER_DIAG_PIN != ULTRASONIC_RIGHT_TRIGGER_PIN,             "Pin conflict: BUZZER_DIAG / RIGHT_TRIG");
static_assert(BUZZER_DIAG_PIN != ULTRASONIC_RIGHT_ECHO_PIN,                "Pin conflict: BUZZER_DIAG / RIGHT_ECHO");

#endif
