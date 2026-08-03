/*
 * comm.h – CommunicationManager
 *
 * Responsibility:
 *   Reads newline-terminated JSON commands from USB Serial, validates them,
 *   and exposes the results to RobotController.
 *   Transmits JSON ACK and telemetry lines back over USB Serial.
 *   Detects communication timeout; sets _connected = false so RobotController
 *   can act on that event.
 *
 *   CommunicationManager does NOT control motors.
 *   CommunicationManager does NOT access ultrasonic sensors.
 *   All motor actions and mode-transition side-effects are performed by
 *   RobotController, which reads state from this class each loop iteration.
 *
 * Transport:
 *   USB Serial only.  Baud rate is defined by SERIAL_BAUD_RATE in config.h.
 *   This is the sole active communication path.
 *
 * Protocol – BridgeRPC JSON over Serial:
 *   Each command is a single JSON object followed by '\n'.
 *   Each response is a single JSON object followed by '\n'.
 *   Input lines longer than COMM_MAX_LINE_LEN bytes are discarded safely.
 *
 * Accepted commands (JSON "cmd" field):
 *   ping       – connectivity test; responds {"type":"pong"}
 *   led        – {"cmd":"led","value":"red"|"off"}
 *                responds {"type":"error","error":"led_not_supported"}
 *   buzzer     – {"cmd":"buzzer","value":"on"|"off"}
 *                responds {"type":"error","error":"buzzer_not_supported"}
 *   motor      – {"cmd":"motor","direction":"forward"|"backward"|"left"|"right"|"stop","speed":<0-255>}
 *                responds {"type":"motor_ack","direction":"<dir>","speed":<spd>}
 *                FORWARD, BACKWARD, LEFT, RIGHT are rejected in AUTO mode with
 *                  {"type":"error","error":"CMD_NOT_ALLOWED_IN_AUTO"}
 *                STOP is always accepted as a safety command regardless of mode;
 *                  it forces MODE_MANUAL and clears any pending movement
 *   mode       – {"cmd":"mode","value":"auto"|"manual"}
 *                responds {"type":"mode_ack","mode":"<value>"}
 *   safe_reset – {"cmd":"safe_reset"}
 *                stops motors, forces MANUAL; responds {"type":"safe_reset_ack","status":"ok","mode":"manual"}
 *
 * Telemetry emitted by sendSensorData() / sendStatus():
 *   {"type":"sensor_data","front":<f>,"rear":<r>,"left":<l>,"right":<r>}
 *   {"type":"motor_status","state":"<navstate>"}
 *
 * State exposed to RobotController:
 *   getMode()            – current OperatingMode (AUTO or MANUAL)
 *   isConnected()        – true while commands are arriving within timeout window
 *   hasPendingMove()     – true when a movement command is waiting
 *   consumePendingMove() – returns and clears the pending movement command
 *
 * Connection loss / timeout:
 *   If no byte is received within SERIAL_CMD_TIMEOUT_MS, this class sets
 *   _connected = false.  RobotController detects this via isConnected() and
 *   stops motors.  _connected becomes true again on the next valid command.
 *
 * Lifecycle:
 *   Call begin() once from setup().
 *   Call update() once per main loop iteration.
 *
 * No external library dependency beyond the Arduino core.
 * No WiFiS3.  No TCP.  No network credentials.
 */

#ifndef COMM_H
#define COMM_H

#include <Arduino.h>
#include "config.h"

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
    char     _buf[COMM_MAX_LINE_LEN + 1];
    uint8_t  _bufLen;
    bool     _overflow;

    OperatingMode _mode;
    MovementCmd   _pendingMove;
    int           _pendingSpeed;
    bool          _hasPending;

    unsigned long _lastRxTime;
    bool          _connected;

    // -----------------------------------------------------------------------
    // JSON helpers – minimal hand-rolled parser (no ArduinoJson dependency)
    // -----------------------------------------------------------------------

    /*
     * jsonGetString() – extract the value of a string key from a flat JSON object.
     * Writes at most `maxLen` characters into `out` and NUL-terminates.
     * Returns true if the key was found.
     */
    static bool jsonGetString(const char* json, const char* key, char* out, uint8_t maxLen) {
        out[0] = '\0';
        char search[40];
        snprintf(search, sizeof(search), "\"%s\"", key);
        const char* p = strstr(json, search);
        if (!p) return false;
        p += strlen(search);
        while (*p == ' ' || *p == ':' || *p == ' ') p++;
        if (*p != '"') return false;
        p++;
        uint8_t i = 0;
        while (*p && *p != '"' && i < maxLen) {
            out[i++] = *p++;
        }
        out[i] = '\0';
        return true;
    }

    /*
     * jsonGetStrictInt() – extract a strict JSON integer value for a key.
     *
     * Accepts:  optional leading '-' followed by one or more decimal digits,
     *           with no subsequent non-digit characters before the JSON token
     *           boundary (space, comma, '}', '\0').
     * Rejects:  decimal fractions (contains '.'), quoted strings ('"'),
     *           bare words (true/false/null, starts with letter), arrays/objects,
     *           numeric prefixes such as "12abc", and empty values.
     *
     * Returns true only when the key is present AND the value is a strict integer.
     * When the key is absent, returns false without touching *out.
     * When the key is present but the value is malformed, returns false and sets
     * *malformed = true so the caller can distinguish absent from invalid.
     */
    static bool jsonGetStrictInt(const char* json, const char* key, int* out, bool* malformed) {
        *malformed = false;
        char search[40];
        snprintf(search, sizeof(search), "\"%s\"", key);
        const char* p = strstr(json, search);
        if (!p) return false;
        p += strlen(search);
        while (*p == ' ' || *p == ':') p++;

        *malformed = true;

        if (*p == '"' || *p == '[' || *p == '{') return false;
        if (*p != '-' && !(*p >= '0' && *p <= '9')) return false;

        bool negative = false;
        if (*p == '-') { negative = true; p++; }
        if (!(*p >= '0' && *p <= '9')) return false;

        long acc = 0;
        bool hasDigit = false;
        while (*p >= '0' && *p <= '9') {
            hasDigit = true;
            int digit = *p - '0';
            if (acc > (300 - digit) / 10) return false;
            acc = acc * 10 + digit;
            p++;
        }
        if (!hasDigit) return false;

        if (*p == '.' || *p == 'e' || *p == 'E') return false;
        if (*p != '\0' && *p != ' ' && *p != ',' && *p != '}' && *p != ']') return false;

        *malformed = false;
        *out = (int)(negative ? -acc : acc);
        return true;
    }

    // -----------------------------------------------------------------------
    // Response emitters
    // -----------------------------------------------------------------------

    static void sendLine(const char* msg) {
        Serial.println(msg);
    }

    static void sendError(const char* code) {
        char buf[64];
        snprintf(buf, sizeof(buf), "{\"type\":\"error\",\"error\":\"%s\"}", code);
        sendLine(buf);
    }

    // -----------------------------------------------------------------------
    // Command dispatch
    // -----------------------------------------------------------------------

    void processLine() {
        if (_overflow) {
            _overflow = false;
            _bufLen   = 0;
            sendError("CMD_TOO_LONG");
            return;
        }

        _buf[_bufLen] = '\0';
        _bufLen = 0;

        if (_buf[0] == '\0') return;

        char cmd[24] = "";
        if (!jsonGetString(_buf, "cmd", cmd, sizeof(cmd) - 1)) {
            sendError("MISSING_CMD");
            return;
        }

        _connected  = true;
        _lastRxTime = millis();

        if (strcmp(cmd, "ping") == 0) {
            sendLine("{\"type\":\"pong\"}");

        } else if (strcmp(cmd, "led") == 0) {
            sendError("led_not_supported");

        } else if (strcmp(cmd, "buzzer") == 0) {
            sendError("buzzer_not_supported");

        } else if (strcmp(cmd, "motor") == 0) {
            char dir[16] = "";
            int  spd     = MOTOR_SPEED_DEFAULT;
            if (!jsonGetString(_buf, "direction", dir, sizeof(dir) - 1)) {
                sendError("MISSING_DIRECTION");
                return;
            }
            bool spdMalformed = false;
            bool hasSpdField  = jsonGetStrictInt(_buf, "speed", &spd, &spdMalformed);
            if (spdMalformed) {
                sendError("INVALID_SPEED");
                return;
            }
            if (hasSpdField && (spd < MOTOR_SPEED_MIN || spd > MOTOR_SPEED_MAX)) {
                sendError("INVALID_SPEED");
                return;
            }

            MovementCmd mc = MOVE_NONE;
            if      (strcmp(dir, "forward")  == 0) mc = MOVE_FORWARD;
            else if (strcmp(dir, "backward") == 0) mc = MOVE_BACKWARD;
            else if (strcmp(dir, "left")     == 0) mc = MOVE_LEFT;
            else if (strcmp(dir, "right")    == 0) mc = MOVE_RIGHT;
            else if (strcmp(dir, "stop")     == 0) mc = MOVE_STOP;
            else { sendError("UNKNOWN_DIRECTION"); return; }

            if (_mode == MODE_AUTO && mc != MOVE_STOP) {
                sendError("CMD_NOT_ALLOWED_IN_AUTO");
                return;
            }

            if (mc == MOVE_STOP) {
                _mode         = MODE_MANUAL;
                _pendingMove  = MOVE_STOP;
                _pendingSpeed = MOTOR_SPEED_DEFAULT;
                _hasPending   = true;
            } else {
                _pendingMove  = mc;
                _pendingSpeed = spd;
                _hasPending   = true;
            }

            char buf[80];
            snprintf(buf, sizeof(buf),
                     "{\"type\":\"motor_ack\",\"direction\":\"%s\",\"speed\":%d}",
                     dir, spd);
            sendLine(buf);

        } else if (strcmp(cmd, "mode") == 0) {
            char value[12] = "";
            if (!jsonGetString(_buf, "value", value, sizeof(value) - 1)) {
                sendError("MISSING_VALUE");
                return;
            }
            if (strcmp(value, "auto") == 0) {
                _mode         = MODE_AUTO;
                _pendingMove  = MOVE_NONE;
                _pendingSpeed = MOTOR_SPEED_DEFAULT;
                _hasPending   = false;
            } else if (strcmp(value, "manual") == 0) {
                if (_mode != MODE_MANUAL) {
                    _pendingMove  = MOVE_STOP;
                    _pendingSpeed = MOTOR_SPEED_DEFAULT;
                    _hasPending   = true;
                }
                _mode = MODE_MANUAL;
            } else {
                sendError("UNKNOWN_MODE");
                return;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), "{\"type\":\"mode_ack\",\"mode\":\"%s\"}", value);
            sendLine(buf);

        } else if (strcmp(cmd, "safe_reset") == 0) {
            _mode         = MODE_MANUAL;
            _pendingMove  = MOVE_STOP;
            _pendingSpeed = MOTOR_SPEED_DEFAULT;
            _hasPending   = true;
            sendLine("{\"type\":\"safe_reset_ack\",\"status\":\"ok\",\"mode\":\"manual\"}");

        } else {
            sendError("UNKNOWN_CMD");
        }
    }

public:
    CommunicationManager()
        : _bufLen(0), _overflow(false),
          _mode(MODE_MANUAL),
          _pendingMove(MOVE_NONE), _pendingSpeed(MOTOR_SPEED_DEFAULT), _hasPending(false),
          _lastRxTime(0),
          _connected(false) {
        _buf[0] = '\0';
    }

    /*
     * begin() – called once from setup() after Serial.begin().
     */
    void begin() {
        _lastRxTime = millis();
        Serial.println("{\"type\":\"ready\"}");
    }

    /*
     * update() – call once per main loop iteration.
     *
     * Reads available bytes from Serial, assembles newline-terminated JSON
     * lines, and dispatches each complete command.
     * Detects timeout and marks _connected = false if no byte arrives within
     * SERIAL_CMD_TIMEOUT_MS.
     */
    void update() {
        if (_connected && millis() - _lastRxTime > SERIAL_CMD_TIMEOUT_MS) {
            _connected = false;
        }

        while (Serial.available()) {
            char c = (char)Serial.read();
            _lastRxTime = millis();

            if (c == '\n') {
                processLine();
            } else if (c == '\r') {
                // ignore CR
            } else {
                if (_bufLen < COMM_MAX_LINE_LEN) {
                    _buf[_bufLen++] = c;
                } else {
                    _overflow = true;
                }
            }
        }
    }

    /*
     * sendStatus() – emit a motor_status telemetry JSON line.
     */
    void sendStatus(const char* state) {
        char buf[64];
        snprintf(buf, sizeof(buf), "{\"type\":\"motor_status\",\"state\":\"%s\"}", state);
        sendLine(buf);
    }

    /*
     * sendSensorData() – emit ultrasonic distances as a sensor_data JSON line.
     * Distances are in cm.
     */
    void sendSensorData(float front, float rear, float left, float right) {
        char buf[96];
        snprintf(buf, sizeof(buf),
                 "{\"type\":\"sensor_data\",\"front\":%.1f,\"rear\":%.1f,\"left\":%.1f,\"right\":%.1f}",
                 (double)front, (double)rear, (double)left, (double)right);
        sendLine(buf);
    }

    OperatingMode getMode()        const { return _mode; }
    bool          isConnected()    const { return _connected; }
    bool          hasPendingMove() const { return _hasPending; }

    /*
     * consumePendingMove() – return and clear the pending movement command.
     * Called only by RobotController.
     */
    MovementCmd consumePendingMove() {
        _hasPending   = false;
        MovementCmd cmd = _pendingMove;
        _pendingMove  = MOVE_NONE;
        _pendingSpeed = MOTOR_SPEED_DEFAULT;
        return cmd;
    }

    /*
     * getPendingSpeed() – return the speed value stored with the last motor command.
     * Must be called before or alongside consumePendingMove().
     */
    int getPendingSpeed() const { return _pendingSpeed; }

    /*
     * resetToManualSafeState() – called by RobotController on timeout.
     * Resets mode to MANUAL and clears any pending movement.
     */
    void resetToManualSafeState() {
        _mode         = MODE_MANUAL;
        _pendingMove  = MOVE_NONE;
        _pendingSpeed = MOTOR_SPEED_DEFAULT;
        _hasPending   = false;
    }
};

#endif
