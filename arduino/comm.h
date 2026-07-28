/*
 * comm.h – CommunicationManager
 *
 * Responsibility:
 *   Accepts a single TCP client connection over Wi-Fi, reads newline-terminated
 *   ASCII commands, validates them, and exposes the results to RobotController.
 *   Transmits plain-text status and sensor lines back to the connected client.
 *   Detects communication timeout and connection loss; sets _connected = false
 *   so RobotController can act on that event.
 *
 *   CommunicationManager does NOT control motors.
 *   CommunicationManager does NOT access ultrasonic sensors.
 *   All motor actions and mode-transition side-effects are performed by
 *   RobotController, which reads state from this class each loop iteration.
 *
 * Transport:
 *   Wi-Fi TCP only.  USB Serial is used for debug diagnostics only.
 *   Primary runtime control path is Wi-Fi.
 *
 * Protocol:
 *   Each command is a single ASCII word followed by '\n' (or "\r\n").
 *   Commands are accepted case-insensitively.
 *   Whitespace (space, tab, CR) is trimmed before matching.
 *   Unknown commands produce "ERROR:UNKNOWN_CMD\n".
 *   Input lines longer than COMM_MAX_CMD_LEN bytes are discarded safely.
 *
 * Accepted commands:
 *   FORWARD   – store pending movement FORWARD  (rejected in AUTO mode)
 *   BACKWARD  – store pending movement BACKWARD (rejected in AUTO mode)
 *   LEFT      – store pending movement LEFT     (rejected in AUTO mode)
 *   RIGHT     – store pending movement RIGHT    (rejected in AUTO mode)
 *   STOP      – store pending MOVE_STOP; switch mode to MANUAL
 *   AUTO      – switch mode to AUTO
 *   MANUAL    – store pending MOVE_STOP (if not already MANUAL); switch mode to MANUAL
 *
 * State exposed to RobotController:
 *   getMode()            – current OperatingMode (AUTO or MANUAL)
 *   isConnected()        – true while a live TCP client is present
 *   hasPendingMove()     – true when a movement command is waiting
 *   consumePendingMove() – returns and clears the pending movement command
 *
 * Connection loss / timeout:
 *   On TCP disconnect or receive timeout (WIFI_CMD_TIMEOUT_MS), this class
 *   sets _connected = false and closes the socket.  RobotController detects
 *   this via isConnected() and performs the motor stop and mode transition.
 *
 * Server lifecycle:
 *   Call begin() once from setup() after WiFi.begin() succeeds.
 *   Call update() once per main loop iteration.
 *
 * No ArduinoJson dependency.  No third-party libraries beyond WiFiS3.
 */

#ifndef COMM_H
#define COMM_H

#include <WiFiS3.h>
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
    WiFiServer  _server;
    WiFiClient  _client;

    char        _buf[COMM_MAX_CMD_LEN + 1];
    uint8_t     _bufLen;
    bool        _overflow;

    OperatingMode _mode;
    MovementCmd   _pendingMove;
    bool          _hasPending;

    unsigned long _lastRxTime;
    bool          _connected;

    // -----------------------------------------------------------------------
    // String helpers
    // -----------------------------------------------------------------------
    static bool isWhitespace(char c) {
        return c == ' ' || c == '\t' || c == '\r';
    }

    static void toUpperInPlace(char* s) {
        for (uint8_t i = 0; s[i]; i++) {
            if (s[i] >= 'a' && s[i] <= 'z') s[i] -= 32;
        }
    }

    static char* trimLeading(char* s) {
        while (*s && isWhitespace(*s)) s++;
        return s;
    }

    static void trimTrailing(char* s) {
        int len = strlen(s);
        while (len > 0 && isWhitespace(s[len - 1])) {
            s[--len] = '\0';
        }
    }

    // -----------------------------------------------------------------------
    // Transport
    // -----------------------------------------------------------------------
    void sendLine(const char* msg) {
        if (_client && _client.connected()) {
            _client.println(msg);
        }
        Serial.println(msg);  // debug mirror
    }

    // -----------------------------------------------------------------------
    // Connection loss handler
    // -----------------------------------------------------------------------
    /*
     * handleDisconnect() – marks the connection as lost and cleans up the
     * socket and receive buffer.
     *
     * Does NOT stop motors or change mode.  RobotController detects
     * _connected == false via isConnected() and performs those actions.
     */
    void handleDisconnect() {
        if (_connected) {
            Serial.println("WIFI:CLIENT_DISCONNECTED");
        }
        _connected  = false;
        _bufLen     = 0;
        _overflow   = false;
        _client.stop();
    }

    // -----------------------------------------------------------------------
    // Command dispatch
    // -----------------------------------------------------------------------
    void processLine() {
        if (_overflow) {
            _overflow = false;
            _bufLen   = 0;
            sendLine("ERROR:CMD_TOO_LONG");
            return;
        }

        _buf[_bufLen] = '\0';
        char* token = trimLeading(_buf);
        trimTrailing(token);
        _bufLen = 0;

        if (token[0] == '\0') return;

        toUpperInPlace(token);

        if (strcmp(token, "FORWARD") == 0) {
            if (_mode == MODE_AUTO) {
                sendLine("ERROR:CMD_NOT_ALLOWED_IN_AUTO");
            } else {
                _pendingMove = MOVE_FORWARD;
                _hasPending  = true;
                sendLine("OK:FORWARD");
            }
        } else if (strcmp(token, "BACKWARD") == 0) {
            if (_mode == MODE_AUTO) {
                sendLine("ERROR:CMD_NOT_ALLOWED_IN_AUTO");
            } else {
                _pendingMove = MOVE_BACKWARD;
                _hasPending  = true;
                sendLine("OK:BACKWARD");
            }
        } else if (strcmp(token, "LEFT") == 0) {
            if (_mode == MODE_AUTO) {
                sendLine("ERROR:CMD_NOT_ALLOWED_IN_AUTO");
            } else {
                _pendingMove = MOVE_LEFT;
                _hasPending  = true;
                sendLine("OK:LEFT");
            }
        } else if (strcmp(token, "RIGHT") == 0) {
            if (_mode == MODE_AUTO) {
                sendLine("ERROR:CMD_NOT_ALLOWED_IN_AUTO");
            } else {
                _pendingMove = MOVE_RIGHT;
                _hasPending  = true;
                sendLine("OK:RIGHT");
            }
        } else if (strcmp(token, "STOP") == 0) {
            _pendingMove = MOVE_STOP;
            _hasPending  = true;
            _mode        = MODE_MANUAL;
            sendLine("OK:STOP");
        } else if (strcmp(token, "AUTO") == 0) {
            _pendingMove = MOVE_NONE;
            _hasPending  = false;
            _mode        = MODE_AUTO;
            sendLine("OK:AUTO");
        } else if (strcmp(token, "MANUAL") == 0) {
            if (_mode != MODE_MANUAL) {
                _pendingMove = MOVE_STOP;
                _hasPending  = true;
                _mode        = MODE_MANUAL;
            }
            sendLine("OK:MANUAL");
        } else {
            sendLine("ERROR:UNKNOWN_CMD");
        }
    }

public:
    /*
     * Constructor.
     * Default mode is MANUAL (safe until a client connects and commands AUTO).
     * No MotorController dependency.
     */
    CommunicationManager()
        : _server(WIFI_SERVER_PORT),
          _bufLen(0), _overflow(false),
          _mode(MODE_MANUAL),
          _pendingMove(MOVE_NONE), _hasPending(false),
          _lastRxTime(0),
          _connected(false) {
        _buf[0] = '\0';
    }

    /*
     * begin() – start the TCP server.
     * Call once from setup(), after WiFi association succeeds.
     */
    void begin() {
        _server.begin();
        _lastRxTime = millis();
        Serial.print("WIFI:SERVER_STARTED port=");
        Serial.println(WIFI_SERVER_PORT);
    }

    /*
     * update() – call once per main loop iteration.
     *
     * Accepts new client connections, reads available bytes, assembles lines,
     * dispatches commands, and detects timeout / TCP close.
     *
     * Does NOT stop motors.  RobotController acts on isConnected() and
     * consumePendingMove() each iteration.
     */
    void update() {
        if (!_connected) {
            WiFiClient incoming = _server.available();
            if (incoming) {
                _client     = incoming;
                _connected  = true;
                _lastRxTime = millis();
                _bufLen     = 0;
                _overflow   = false;
                Serial.print("WIFI:CLIENT_CONNECTED ip=");
                Serial.println(_client.remoteIP());
                sendLine("READY");
            }
            return;
        }

        if (!_client.connected()) {
            handleDisconnect();
            return;
        }

        if (millis() - _lastRxTime > WIFI_CMD_TIMEOUT_MS) {
            Serial.println("WIFI:TIMEOUT");
            handleDisconnect();
            return;
        }

        while (_client.available()) {
            char c = (char)_client.read();
            _lastRxTime = millis();

            if (c == '\n') {
                processLine();
            } else if (c == '\r') {
                // ignore CR; '\n' triggers processing
            } else {
                if (_bufLen < COMM_MAX_CMD_LEN) {
                    _buf[_bufLen++] = c;
                } else {
                    _overflow = true;
                }
            }
        }
    }

    /*
     * sendStatus() – transmit a plain-text status line to the connected client.
     * Call from RobotController at COMM_SEND_INTERVAL_MS intervals.
     */
    void sendStatus(const char* status) {
        sendLine(status);
    }

    /*
     * sendSensorData() – transmit ultrasonic distances as a comma-separated line.
     * Format: "SENSOR:front,rear,left,right\n"
     * Distances are in cm, rounded to one decimal place.
     */
    void sendSensorData(float front, float rear, float left, float right) {
        char buf[64];
        snprintf(buf, sizeof(buf), "SENSOR:%.1f,%.1f,%.1f,%.1f",
                 (double)front, (double)rear, (double)left, (double)right);
        sendLine(buf);
    }

    OperatingMode getMode()        const { return _mode; }
    bool          isConnected()    const { return _connected; }
    bool          hasPendingMove() const { return _hasPending; }

    /*
     * consumePendingMove() – return and clear the pending movement command.
     * Called by RobotController; must not be called by any other module.
     */
    MovementCmd consumePendingMove() {
        _hasPending  = false;
        MovementCmd cmd = _pendingMove;
        _pendingMove = MOVE_NONE;
        return cmd;
    }

    /*
     * forceManual() – called by RobotController when a disconnect or timeout
     * is detected, to reset mode to MANUAL and clear any pending movement.
     * Motor stop is performed by RobotController before calling this.
     */
    void resetToManualSafeState() {
        _mode        = MODE_MANUAL;
        _pendingMove = MOVE_NONE;
        _hasPending  = false;
    }
};

#endif
