/*
 * comm.h – CommunicationManager
 *
 * Responsibility:
 *   Owns MCU command state for the UNO Q RouterBridge/RPClite transport.
 *   Arduino_RouterBridge callbacks delegate to this class; RobotController
 *   remains the only owner of motor actions.
 *
 * Transport:
 *   App Lab Python calls explicit Bridge RPC endpoints:
 *     asm_ping
 *     asm_led
 *     asm_buzzer
 *     asm_motor
 *     asm_mode
 *     asm_safe_reset
 *     asm_control_tick
 *     asm_manual_hold
 *     asm_status
 *     asm_ultrasonic_diag
 *     asm_buzzer_tone_diag
 *
 * Safety:
 *   Motion Lease remains MCU-authoritative.  ping never starts or renews the
 *   lease.  control_tick renews only an already active lease.  STOP and
 *   safe_reset force MANUAL and queue a STOP for RobotController to consume.
 */

#ifndef COMM_H
#define COMM_H

#include <Arduino.h>
#include "config.h"
#include "pins.h"

enum OperatingMode {
    MODE_AUTO,
    MODE_MANUAL
};

enum MovementCmd {
    MOVE_NONE,
    MOVE_FORWARD,
    MOVE_BACKWARD,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_STOP
};

class CommunicationManager {
private:
    OperatingMode _mode;
    MovementCmd   _pendingMove;
    int           _pendingSpeed;
    bool          _hasPending;
    bool          _stopLatched;

    bool          _motionLeaseActive;
    unsigned long _lastMotionLeaseTime;
    bool          _manualHoldActive;
    unsigned long _lastManualHoldTime;

    bool          _warningActive;
    unsigned long _warningStartTime;
    unsigned long _lastLedToggleTime;
    bool          _ledOn;
    bool          _buzzerOn;
    unsigned long _lastBuzzerToggleTime;

    unsigned long _lastCommandTime;
    bool          _connected;

    void _startMotionLease() {
        _motionLeaseActive   = true;
        _lastMotionLeaseTime = millis();
    }

    void _renewMotionLease() {
        _lastMotionLeaseTime = millis();
    }

    void _clearMotionLease() {
        _motionLeaseActive   = false;
        _lastMotionLeaseTime = 0;
    }

    void _startManualHold() {
        _manualHoldActive   = true;
        _lastManualHoldTime = millis();
    }

    void _renewManualHold() {
        _lastManualHoldTime = millis();
    }

    void _clearManualHold() {
        _manualHoldActive   = false;
        _lastManualHoldTime = 0;
    }

    bool _motionLeaseExpired() const {
        if (!_motionLeaseActive) return false;
        return (millis() - _lastMotionLeaseTime) > MOTION_LEASE_TIMEOUT_MS;
    }

    bool _manualHoldExpired() const {
        if (!_manualHoldActive) return false;
        return (millis() - _lastManualHoldTime) > MANUAL_HOLD_TIMEOUT_MS;
    }

    void _expireLease() {
        // Lease expiry is converted into the same pending STOP path used by explicit stop commands.
        _clearMotionLease();
        _clearManualHold();
        _mode         = MODE_MANUAL;
        _pendingMove  = MOVE_STOP;
        _pendingSpeed = MOTOR_SPEED_DEFAULT;
        _hasPending   = true;
        _stopLatched  = true;
    }

    void _setLed(bool on) {
        _ledOn = on;
        digitalWrite(ALERT_LED_PIN, on ? HIGH : LOW);
    }

    void _setBuzzer(bool on) {
        _buzzerOn = on;
        if (on) {
            tone(BUZZER_DIAG_PIN, BUZZER_WARN_FREQ_HZ);
        } else {
            noTone(BUZZER_DIAG_PIN);
            digitalWrite(BUZZER_DIAG_PIN, LOW);
        }
    }

    void _startWarning() {
        _warningActive     = true;
        _warningStartTime  = millis();
        _lastLedToggleTime = _warningStartTime;
        _mode              = MODE_MANUAL;
        _pendingMove       = MOVE_STOP;
        _pendingSpeed      = MOTOR_SPEED_DEFAULT;
        _hasPending        = true;
        _stopLatched       = true;
        _clearMotionLease();
        _clearManualHold();
        _setLed(true);
        _lastBuzzerToggleTime = _warningStartTime;
        _setBuzzer(true);
    }

    void _refreshWarning() {
        _warningStartTime  = millis();
        _lastLedToggleTime = _warningStartTime;
        _clearMotionLease();
        _clearManualHold();
        _setLed(true);
    }

    void _clearWarningLed() {
        _warningActive     = false;
        _warningStartTime  = 0;
        _lastLedToggleTime = 0;
        _setLed(false);
        _setBuzzer(false);
        _lastBuzzerToggleTime = 0;
    }

    void _markCommandReceived() {
        _connected = true;
        _lastCommandTime = millis();
    }

    String _error(const char* code) const {
        String out = "{\"type\":\"error\",\"error\":\"";
        out += code;
        out += "\"}";
        return out;
    }

    String _motorAck(const char* direction, int speed) const {
        String out = "{\"type\":\"motor_ack\",\"direction\":\"";
        out += direction;
        out += "\",\"speed\":";
        out += String(speed);
        out += "}";
        return out;
    }

    MovementCmd _movementFromDirection(const String& dir) const {
        if (dir == "forward")  return MOVE_FORWARD;
        if (dir == "backward") return MOVE_BACKWARD;
        if (dir == "left")     return MOVE_LEFT;
        if (dir == "right")    return MOVE_RIGHT;
        if (dir == "stop")     return MOVE_STOP;
        return MOVE_NONE;
    }

    const char* _modeString() const {
        return _mode == MODE_AUTO ? "auto" : "manual";
    }

    void _appendDistance(String& out, bool available, float distance) const {
        if (available) {
            out += String(distance, 1);
        } else {
            out += "null";
        }
    }

public:
    CommunicationManager()
        : _mode(MODE_MANUAL),
          _pendingMove(MOVE_NONE), _pendingSpeed(MOTOR_SPEED_DEFAULT), _hasPending(false),
          _stopLatched(false),
          _motionLeaseActive(false), _lastMotionLeaseTime(0),
          _manualHoldActive(false), _lastManualHoldTime(0),
          _warningActive(false), _warningStartTime(0), _lastLedToggleTime(0), _ledOn(false),
          _buzzerOn(false), _lastBuzzerToggleTime(0),
          _lastCommandTime(0), _connected(false) {}

    void begin() {
        pinMode(ALERT_LED_PIN, OUTPUT);
        pinMode(BUZZER_DIAG_PIN, OUTPUT);
        noTone(BUZZER_DIAG_PIN);
        digitalWrite(BUZZER_DIAG_PIN, LOW);
        _clearWarningLed();
        _lastCommandTime = millis();
    }

    void update() {
        if (_connected && millis() - _lastCommandTime > COMMAND_TIMEOUT_MS) {
            _connected = false;
            _clearMotionLease();
            _clearManualHold();
            _clearWarningLed();
        }
        if (_mode == MODE_MANUAL && _manualHoldExpired()) {
            _expireLease();
        }
        if (_motionLeaseExpired()) {
            _expireLease();
        }
        updateWarning();
    }

    String rpcPing() {
        _markCommandReceived();
        return "{\"type\":\"pong\"}";
    }

    String rpcLed(String value) {
        _markCommandReceived();
        if (value == "red") {
            if (!_warningActive) {
                _startWarning();
            } else {
                _refreshWarning();
            }
        } else if (value == "off") {
            _clearWarningLed();
        } else {
            return _error("UNKNOWN_LED");
        }

        String out = "{\"type\":\"led_ack\",\"color\":\"";
        out += value;
        out += "\"}";
        return out;
    }

    String rpcBuzzer(String command) {
        _markCommandReceived();
        if (command == "test") {
            if (_warningActive) {
                return _error("WARNING_ACTIVE");
            }
            noTone(BUZZER_DIAG_PIN);
            digitalWrite(BUZZER_DIAG_PIN, LOW);
            tone(BUZZER_DIAG_PIN, BUZZER_WARN_FREQ_HZ, BUZZER_TEST_DURATION_MS);
            delay(BUZZER_TEST_DURATION_MS);  // ponytail: bounded 250ms; MCU blocks during RPC anyway
            noTone(BUZZER_DIAG_PIN);
            digitalWrite(BUZZER_DIAG_PIN, LOW);
            return "{\"type\":\"buzzer_ack\",\"state\":\"test\",\"ok\":true}";
        }
        noTone(BUZZER_DIAG_PIN);
        digitalWrite(BUZZER_DIAG_PIN, LOW);
        return _error("UNKNOWN_BUZZER_COMMAND");
    }

    String rpcBuzzerToneDiag(int frequency_hz, int duration_ms) {
        _markCommandReceived();
        noTone(BUZZER_DIAG_PIN);
        digitalWrite(BUZZER_DIAG_PIN, LOW);
        if (frequency_hz < BUZZER_TONE_DIAG_FREQUENCY_MIN_HZ || frequency_hz > BUZZER_TONE_DIAG_FREQUENCY_MAX_HZ) {
            return _error("INVALID_FREQUENCY");
        }
        if (duration_ms < BUZZER_DIAG_DURATION_MIN_MS || duration_ms > BUZZER_DIAG_DURATION_MAX_MS) {
            return _error("INVALID_DURATION");
        }
        tone(BUZZER_DIAG_PIN, frequency_hz, duration_ms);
        delay(duration_ms);
        noTone(BUZZER_DIAG_PIN);
        digitalWrite(BUZZER_DIAG_PIN, LOW);
        return "{\"type\":\"buzzer_tone_diag\",\"status\":\"ok\",\"pin\":" + String(BUZZER_DIAG_PIN)
               + ",\"frequency_hz\":" + String(frequency_hz)
               + ",\"duration_ms\":" + String(duration_ms) + "}";
    }

    String rpcMotor(String direction, int speed) {
        _markCommandReceived();
        int spd = speed;
        if (spd < MOTOR_SPEED_MIN || spd > MOTOR_SPEED_MAX) {
            return _error("INVALID_SPEED");
        }

        MovementCmd mc = _movementFromDirection(direction);
        if (mc == MOVE_NONE) {
            return _error("UNKNOWN_DIRECTION");
        }

        if (_mode == MODE_AUTO && mc != MOVE_STOP) {
            return _error("CMD_NOT_ALLOWED_IN_AUTO");
        }

        // STOP is always accepted because it is the safety path that clears active motion authority.
        if (mc == MOVE_STOP) {
            _mode         = MODE_MANUAL;
            _pendingMove  = MOVE_STOP;
            _pendingSpeed = MOTOR_SPEED_DEFAULT;
            _hasPending   = true;
            _stopLatched  = true;
            _clearMotionLease();
            _clearManualHold();
        } else if (_warningActive) {
            return _error("CMD_BLOCKED_BY_WARNING");
        } else if (_stopLatched) {
            return _error("CMD_BLOCKED_BY_STOP_LATCH");
        } else if (!_stopLatched && !_warningActive) {
            _pendingMove  = mc;
            _pendingSpeed = spd;
            _hasPending   = true;
            _startMotionLease();
            _startManualHold();
        }

        return _motorAck(direction.c_str(), spd);
    }

    String rpcMode(String value) {
        _markCommandReceived();
        if (value == "auto") {
            if (_warningActive) {
                return _error("CMD_BLOCKED_BY_WARNING");
            }
            if (_stopLatched) {
                return _error("CMD_BLOCKED_BY_STOP_LATCH");
            }
            _mode         = MODE_AUTO;
            _pendingMove  = MOVE_NONE;
            _pendingSpeed = MOTOR_SPEED_DEFAULT;
            _hasPending   = false;
            _clearManualHold();
            _startMotionLease();
        } else if (value == "manual") {
            if (_mode != MODE_MANUAL) {
                _pendingMove  = MOVE_STOP;
                _pendingSpeed = MOTOR_SPEED_DEFAULT;
                _hasPending   = true;
            }
            _mode = MODE_MANUAL;
            _clearMotionLease();
            _clearManualHold();
        } else {
            return _error("UNKNOWN_MODE");
        }

        String out = "{\"type\":\"mode_ack\",\"mode\":\"";
        out += value;
        out += "\"}";
        return out;
    }

    String rpcControlTick() {
        _markCommandReceived();
        if (_mode == MODE_MANUAL && _manualHoldExpired()) {
            _expireLease();
        }
        if (_motionLeaseActive && (_mode == MODE_AUTO || _manualHoldActive)) {
            _renewMotionLease();
            return "{\"type\":\"control_tick_ack\",\"motion_authorized\":true}";
        }
        return "{\"type\":\"control_tick_ack\",\"motion_authorized\":false}";
    }

    String rpcManualHold() {
        _markCommandReceived();
        if (_warningActive) return _error("CMD_BLOCKED_BY_WARNING");
        if (_stopLatched) return _error("CMD_BLOCKED_BY_STOP_LATCH");
        if (_mode != MODE_MANUAL || !_motionLeaseActive || !_manualHoldActive) {
            return "{\"type\":\"manual_hold_ack\",\"active\":false}";
        }
        if (_manualHoldExpired()) {
            _expireLease();
            return "{\"type\":\"manual_hold_ack\",\"active\":false}";
        }
        _renewManualHold();
        _renewMotionLease();
        return "{\"type\":\"manual_hold_ack\",\"active\":true}";
    }

    String rpcSafeReset() {
        _markCommandReceived();
        _clearMotionLease();
        _clearManualHold();
        _mode         = MODE_MANUAL;
        _pendingMove  = MOVE_STOP;
        _pendingSpeed = MOTOR_SPEED_DEFAULT;
        _hasPending   = true;
        _stopLatched  = true;
        _clearWarningLed();
        return "{\"type\":\"safe_reset_ack\",\"status\":\"ok\",\"mode\":\"manual\"}";
    }

    String rpcStatus(
        const char* movement,
        int motorSpeed,
        bool frontAvailable,
        float frontDistance,
        bool rearAvailable,
        float rearDistance,
        bool leftAvailable,
        float leftDistance,
        bool rightAvailable,
        float rightDistance
    ) {
        unsigned long now = millis();
        // Capture freshness before _markCommandReceived() so status reports the pre-request safety state.
        bool commandFresh = _connected && (now - _lastCommandTime <= COMMAND_TIMEOUT_MS);
        bool controlTickFresh = _motionLeaseActive && (now - _lastMotionLeaseTime <= MOTION_LEASE_TIMEOUT_MS);
        _markCommandReceived();

        String out = "{\"type\":\"status\",";
        out += "\"mode\":\"";
        out += _modeString();
        out += "\",\"movement\":\"";
        out += movement;
        out += "\",\"motor_speed\":";
        out += String(motorSpeed);
        out += ",\"motion_lease_active\":";
        out += _motionLeaseActive ? "true" : "false";
        out += ",\"control_tick_fresh\":";
        out += controlTickFresh ? "true" : "false";
        out += ",\"command_fresh\":";
        out += commandFresh ? "true" : "false";
        out += ",\"warning_active\":";
        out += _warningActive ? "true" : "false";
        out += ",\"safety_state\":\"";
        out += _warningActive ? "warning" : ((_motionLeaseActive && commandFresh) ? "motion_authorized" : "safe");
        out += "\",\"distances\":{\"front\":";
        _appendDistance(out, frontAvailable, frontDistance);
        out += ",\"rear\":";
        _appendDistance(out, rearAvailable, rearDistance);
        out += ",\"left\":";
        _appendDistance(out, leftAvailable, leftDistance);
        out += ",\"right\":";
        _appendDistance(out, rightAvailable, rightDistance);
        out += "}}";
        return out;
    }

    String rpcUltrasonicDiagnostic(
        bool frontAvailable,
        float frontDistance,
        bool rearAvailable,
        float rearDistance,
        bool leftAvailable,
        float leftDistance,
        bool rightAvailable,
        float rightDistance
    ) {
        _markCommandReceived();

        String out = "{\"type\":\"ultrasonic_diag\",\"mode\":\"manual\",\"distances\":{\"front\":";
        _appendDistance(out, frontAvailable, frontDistance);
        out += ",\"rear\":";
        _appendDistance(out, rearAvailable, rearDistance);
        out += ",\"left\":";
        _appendDistance(out, leftAvailable, leftDistance);
        out += ",\"right\":";
        _appendDistance(out, rightAvailable, rightDistance);
        out += "}}";
        return out;
    }

    void sendStatus(const char* state) {
        (void)state; // Legacy async hook; current telemetry is exposed through rpcStatus().
    }

    void sendSensorData(float front, float rear, float left, float right) {
        (void)front;
        (void)rear;
        (void)left;
        (void)right; // Legacy async hook; current telemetry is exposed through rpcStatus().
    }

    OperatingMode getMode()         const { return _mode; }
    bool          isConnected()     const { return _connected; }
    bool          hasPendingMove()  const { return _hasPending; }
    bool          isWarningActive() const { return _warningActive; }

    MovementCmd consumePendingMove() {
        // Consuming any pending command also releases the STOP latch for the next explicit command.
        _hasPending   = false;
        _stopLatched  = false;
        MovementCmd cmd = _pendingMove;
        _pendingMove  = MOVE_NONE;
        _pendingSpeed = MOTOR_SPEED_DEFAULT;
        return cmd;
    }

    int getPendingSpeed() const { return _pendingSpeed; }

    void resetToManualSafeState() {
        _clearMotionLease();
        _clearManualHold();
        _clearWarningLed();
        _mode         = MODE_MANUAL;
        _stopLatched  = false;
        _pendingMove  = MOVE_NONE;
        _pendingSpeed = MOTOR_SPEED_DEFAULT;
        _hasPending   = false;
    }

    void updateWarning() {
        if (!_warningActive) return;

        unsigned long now = millis();
        if (now - _warningStartTime >= HELMET_WARNING_BUZZER_DURATION_MS) {
            _clearWarningLed();
            _clearMotionLease();
            _clearManualHold();
            _mode         = MODE_MANUAL;
            _pendingMove  = MOVE_NONE;
            _pendingSpeed = MOTOR_SPEED_DEFAULT;
            _hasPending   = false;
            _stopLatched  = false;
            return;
        }

        if (now - _lastLedToggleTime >= LED_BLINK_INTERVAL_MS) {
            _lastLedToggleTime = now;
            _setLed(!_ledOn);
        }

        if (now - _lastBuzzerToggleTime >= BUZZER_WARN_HALF_PERIOD_MS) {
            _lastBuzzerToggleTime = now;
            _setBuzzer(!_buzzerOn);
        }
    }

    bool isMotionAuthorized() const { return _motionLeaseActive; }
};

#endif
