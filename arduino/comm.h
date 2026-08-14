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

    bool          _warningActive;
    unsigned long _warningStartTime;
    unsigned long _lastLedToggleTime;
    bool          _ledOn;

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

    bool _motionLeaseExpired() const {
        if (!_motionLeaseActive) return false;
        return (millis() - _lastMotionLeaseTime) > MOTION_LEASE_TIMEOUT_MS;
    }

    void _expireLease() {
        _clearMotionLease();
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
        _setLed(true);
    }

    void _refreshWarning() {
        _warningStartTime  = millis();
        _lastLedToggleTime = _warningStartTime;
        _clearMotionLease();
        _setLed(true);
    }

    void _clearWarningLed() {
        _warningActive     = false;
        _warningStartTime  = 0;
        _lastLedToggleTime = 0;
        _setLed(false);
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

public:
    CommunicationManager()
        : _mode(MODE_MANUAL),
          _pendingMove(MOVE_NONE), _pendingSpeed(MOTOR_SPEED_DEFAULT), _hasPending(false),
          _stopLatched(false),
          _motionLeaseActive(false), _lastMotionLeaseTime(0),
          _warningActive(false), _warningStartTime(0), _lastLedToggleTime(0), _ledOn(false),
          _lastCommandTime(0), _connected(false) {}

    void begin() {
        pinMode(ALERT_LED_PIN, OUTPUT);
        _clearWarningLed();
        _lastCommandTime = millis();
    }

    void update() {
        if (_connected && millis() - _lastCommandTime > COMMAND_TIMEOUT_MS) {
            _connected = false;
            _clearMotionLease();
            _clearWarningLed();
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
            if (!_warningActive) {
                _clearWarningLed();
            }
        } else {
            return _error("UNKNOWN_LED");
        }

        String out = "{\"type\":\"led_ack\",\"color\":\"";
        out += value;
        out += "\"}";
        return out;
    }

    String rpcBuzzer(String state) {
        _markCommandReceived();
        (void)state;
        return _error("buzzer_not_supported");
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

        if (mc == MOVE_STOP) {
            _mode         = MODE_MANUAL;
            _pendingMove  = MOVE_STOP;
            _pendingSpeed = MOTOR_SPEED_DEFAULT;
            _hasPending   = true;
            _stopLatched  = true;
            _clearMotionLease();
        } else if (!_stopLatched && !_warningActive) {
            _pendingMove  = mc;
            _pendingSpeed = spd;
            _hasPending   = true;
            _startMotionLease();
        }

        return _motorAck(direction.c_str(), spd);
    }

    String rpcMode(String value) {
        _markCommandReceived();
        if (value == "auto") {
            if (!_stopLatched && !_warningActive) {
                _mode         = MODE_AUTO;
                _pendingMove  = MOVE_NONE;
                _pendingSpeed = MOTOR_SPEED_DEFAULT;
                _hasPending   = false;
                _startMotionLease();
            }
        } else if (value == "manual") {
            if (_mode != MODE_MANUAL) {
                _pendingMove  = MOVE_STOP;
                _pendingSpeed = MOTOR_SPEED_DEFAULT;
                _hasPending   = true;
            }
            _mode = MODE_MANUAL;
            _clearMotionLease();
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
        if (_motionLeaseActive) {
            _renewMotionLease();
            return "{\"type\":\"control_tick_ack\",\"motion_authorized\":true}";
        }
        return "{\"type\":\"control_tick_ack\",\"motion_authorized\":false}";
    }

    String rpcSafeReset() {
        _markCommandReceived();
        _clearMotionLease();
        _mode         = MODE_MANUAL;
        _pendingMove  = MOVE_STOP;
        _pendingSpeed = MOTOR_SPEED_DEFAULT;
        _hasPending   = true;
        _stopLatched  = true;
        _clearWarningLed();
        return "{\"type\":\"safe_reset_ack\",\"status\":\"ok\",\"mode\":\"manual\"}";
    }

    void sendStatus(const char* state) {
        (void)state; // Phase 2 telemetry will expose status through Bridge RPC.
    }

    void sendSensorData(float front, float rear, float left, float right) {
        (void)front;
        (void)rear;
        (void)left;
        (void)right; // Phase 2 telemetry will expose distances through Bridge RPC.
    }

    OperatingMode getMode()         const { return _mode; }
    bool          isConnected()     const { return _connected; }
    bool          hasPendingMove()  const { return _hasPending; }
    bool          isWarningActive() const { return _warningActive; }

    MovementCmd consumePendingMove() {
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
        if (now - _warningStartTime >= WARNING_DURATION_MS) {
            _clearWarningLed();
            _clearMotionLease();
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
    }

    bool isMotionAuthorized() const { return _motionLeaseActive; }
};

#endif
