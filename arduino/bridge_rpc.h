/*
 * Bridge RPC Communication Library
 *
 * This library provides JSON-based RPC (Remote Procedure Call) communication
 * between the Arduino and external systems (Python MPU, App Lab, etc.).
 *
 * Features:
 * - JSON command processing for motor, LED, and buzzer control
 * - Real-time sensor data transmission
 * - Automatic timeout detection and safe mode activation
 * - Bidirectional communication with acknowledgments
 * - Configurable data transmission intervals
 *
 * Communication Protocol:
 * - Commands received via Serial in JSON format
 * - Sensor data and status transmitted as JSON messages
 * - Safe mode triggered on communication timeout
 */

#ifndef BRIDGE_RPC_H
#define BRIDGE_RPC_H

#include <ArduinoJson.h>  // JSON serialization/deserialization library
#include "motor.h"        // Motor controller interface
#include "alert.h"        // Alert system interface

/**
 * BridgeRPC Class
 *
 * Manages JSON-based communication between Arduino and external systems.
 * Handles command processing, sensor data transmission, and safety features.
 */
class BridgeRPC {
private:
    // Timing variables for communication monitoring
    unsigned long lastReceiveTime;  // Timestamp of last received command
    unsigned long lastSendTime;     // Timestamp of last data transmission

    // Communication buffers and JSON documents
    String inputBuffer;                    // Buffer for incoming serial data
    StaticJsonDocument<256> receiveDoc;    // JSON parser for incoming commands
    StaticJsonDocument<256> sendDoc;       // JSON builder for outgoing data

    // Safety and control state
    bool safeMode;                         // Emergency safe mode flag
    String operatingMode;                  // Current operating mode: "auto" or "manual"

    // Motor command state
    bool hasNewMotorCommand;               // Flag indicating new motor command received
    String lastMotorDirection;             // Last received motor direction
    int lastMotorSpeed;                    // Last received motor speed

    // Hardware controller references
    MotorController* motorController;      // Reference to motor controller
    AlertController* alertController;      // Reference to alert controller

    // Configuration constants and variables
    const unsigned long TIMEOUT_THRESHOLD = 500;  // Communication timeout in ms
    unsigned long sendInterval;                    // Data transmission interval

public:
    /**
     * Constructor - Initialize bridge RPC with hardware controllers
     * @param motor: Pointer to motor controller instance
     * @param alert: Pointer to alert controller instance
     */
    BridgeRPC(MotorController* motor, AlertController* alert) {
        // Initialize timing variables
        lastReceiveTime = 0;
        lastSendTime = 0;

        // Initialize communication state
        safeMode = false;
        operatingMode = "auto";             // Default to autonomous mode on startup
        inputBuffer = "";

        // Initialize motor command state
        hasNewMotorCommand = false;
        lastMotorDirection = "";
        lastMotorSpeed = 0;

        // Store hardware controller references
        motorController = motor;
        alertController = alert;

        // Set default data transmission rate (10Hz)
        sendInterval = 100;
    }

    /**
     * Main update method - call regularly in main loop
     * Handles communication timeout checking, command processing,
     * and periodic sensor data transmission
     */
    void update() {
        checkTimeout();      // Monitor for communication timeouts
        receiveCommands();   // Process incoming serial commands

        // Send periodic sensor data at configured interval
        if (millis() - lastSendTime >= sendInterval) {
            sendSensorData();
            lastSendTime = millis();
        }
    }

    /**
     * Monitor communication timeout and activate safe mode if needed
     * Triggers emergency stop if no commands received within threshold
     */
    void checkTimeout() {
        // Check if timeout occurred (only after first command received)
        if (millis() - lastReceiveTime > TIMEOUT_THRESHOLD && lastReceiveTime > 0) {
            if (!safeMode) {
                safeMode = true;     // Enable safe mode
                triggerSafeMode();   // Send timeout notification
            }
        }
    }

    /**
     * Activate safe mode and notify external systems
     * Sends emergency stop message when communication timeout occurs
     */
    void triggerSafeMode() {
        sendDoc.clear();
        sendDoc["status"] = "timeout";         // Indicate timeout condition
        sendDoc["action"] = "emergency_stop";  // Signal emergency stop
        String output;
        serializeJson(sendDoc, output);         // Convert to JSON string
        Serial.println(output);                 // Transmit via serial
    }

    /**
     * Read and buffer incoming serial commands
     * Commands are processed line-by-line (newline terminated)
     */
    void receiveCommands() {
        while (Serial.available()) {
            char c = Serial.read();           // Read single character
            if (c == '\n') {
                processCommand(inputBuffer);   // Process complete command
                inputBuffer = "";              // Clear buffer for next command
            } else {
                inputBuffer += c;              // Add character to buffer
            }
        }
    }

    /**
     * Send a JSON error response with the given error code.
     * @param errorCode: Short string identifying the error
     */
    void sendError(const char* errorCode) {
        sendDoc.clear();
        sendDoc["type"] = "error";
        sendDoc["error"] = errorCode;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Parse and execute JSON commands from external systems.
     * Supports motor control, LED control, buzzer control, ping, and mode.
     * All fields are validated before any state change or ACK is sent.
     * @param command: JSON string containing command data
     */
    void processCommand(String command) {
        DeserializationError error = deserializeJson(receiveDoc, command);
        if (error) {
            sendError("invalid_json");
            return;
        }

        // Validate cmd key presence and type
        if (!receiveDoc.containsKey("cmd")) {
            sendError("missing_cmd");
            return;
        }
        if (!receiveDoc["cmd"].is<const char*>()) {
            sendError("invalid_cmd");
            return;
        }

        String cmd = receiveDoc["cmd"].as<String>();
        if (cmd.length() == 0) {
            sendError("invalid_cmd");
            return;
        }

        // Reject unknown commands before updating timing or safeMode
        if (cmd != "motor" && cmd != "led" && cmd != "buzzer" &&
            cmd != "ping" && cmd != "mode") {
            sendError("unknown_cmd");
            return;
        }

        // Valid known command: update communication timestamp and clear safe mode
        lastReceiveTime = millis();
        safeMode = false;

        // Route to validated handlers
        if (cmd == "motor") {
            // Validate direction field
            if (!receiveDoc.containsKey("direction")) {
                sendError("missing_direction");
                return;
            }
            if (!receiveDoc["direction"].is<const char*>()) {
                sendError("invalid_direction");
                return;
            }
            String direction = receiveDoc["direction"].as<String>();
            if (direction != "forward" && direction != "backward" &&
                direction != "left" && direction != "right" &&
                direction != "stop") {
                sendError("invalid_direction");
                return;
            }

            // Validate speed field
            if (!receiveDoc.containsKey("speed")) {
                sendError("missing_speed");
                return;
            }
            if (!receiveDoc["speed"].is<int>()) {
                sendError("invalid_speed");
                return;
            }
            int speed = receiveDoc["speed"].as<int>();
            if (speed < 0 || speed > 255) {
                sendError("invalid_speed");
                return;
            }

            handleMotorCommand(direction, speed);
        }
        else if (cmd == "led") {
            // Validate value field
            if (!receiveDoc.containsKey("value")) {
                sendError("missing_value");
                return;
            }
            if (!receiveDoc["value"].is<const char*>()) {
                sendError("invalid_led_value");
                return;
            }
            String color = receiveDoc["value"].as<String>();
            if (color != "red" && color != "off") {
                sendError("invalid_led_value");
                return;
            }

            handleLedCommand(color);
        }
        else if (cmd == "buzzer") {
            // Validate value field
            if (!receiveDoc.containsKey("value")) {
                sendError("missing_value");
                return;
            }
            if (!receiveDoc["value"].is<const char*>()) {
                sendError("invalid_buzzer_value");
                return;
            }
            String state = receiveDoc["value"].as<String>();
            if (state != "on" && state != "off") {
                sendError("invalid_buzzer_value");
                return;
            }

            handleBuzzerCommand(state);
        }
        else if (cmd == "ping") {
            sendPongResponse();
        }
        else if (cmd == "mode") {
            // Validate value field
            if (!receiveDoc.containsKey("value")) {
                sendError("missing_value");
                return;
            }
            if (!receiveDoc["value"].is<const char*>()) {
                sendError("invalid_mode");
                return;
            }
            String value = receiveDoc["value"].as<String>();
            if (value != "auto" && value != "manual") {
                sendError("invalid_mode");
                return;
            }

            handleModeCommand(value);
        }
    }

    /**
     * Handle motor movement commands
     * @param direction: Movement direction ("forward", "backward", "left", "right", "stop")
     * @param speed: PWM speed value (0-255)
     */
    void handleMotorCommand(String direction, int speed) {
        // Store motor command for main loop processing
        lastMotorDirection = direction;
        lastMotorSpeed = speed;
        hasNewMotorCommand = true;

        // Send acknowledgment with command details
        sendDoc.clear();
        sendDoc["type"] = "motor_ack";
        sendDoc["direction"] = direction;
        sendDoc["speed"] = speed;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Handle LED control commands
     * @param color: LED command ("red" to start blinking, "off" to turn off)
     */
    void handleLedCommand(String color) {
        // Execute LED command if alert controller is available
        if (alertController) {
            if (color == "red") {
                alertController->startAlert();     // Start LED blinking
            } else if (color == "off") {
                alertController->stopAlert();      // Turn off LED
            }
        }

        // Send acknowledgment with command details
        sendDoc.clear();
        sendDoc["type"] = "led_ack";
        sendDoc["color"] = color;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Handle buzzer control commands
     * @param state: Buzzer state ("on" or "off")
     */
    void handleBuzzerCommand(String state) {
        // Execute buzzer command if alert controller is available
        if (alertController) {
            if (state == "on") {
                alertController->buzzerOn();       // Turn on buzzer
            } else if (state == "off") {
                alertController->buzzerOff();      // Turn off buzzer
            }
        }

        // Send acknowledgment with command details
        sendDoc.clear();
        sendDoc["type"] = "buzzer_ack";
        sendDoc["state"] = state;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Handle operating mode selection commands
     * @param value: Requested mode ("auto" or "manual")
     */
    void handleModeCommand(String value) {
        if (value == "auto" || value == "manual") {
            operatingMode = value;             // Persist the selected mode
        }

        // Send acknowledgment with the active mode
        sendDoc.clear();
        sendDoc["type"] = "mode_ack";
        sendDoc["mode"] = operatingMode;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Send pong response to ping command
     * Used for connectivity testing and latency measurement
     */
    void sendPongResponse() {
        sendDoc.clear();
        sendDoc["type"] = "pong";
        sendDoc["timestamp"] = millis();     // Include current timestamp
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Send periodic sensor data (placeholder)
     * This method is called at regular intervals to transmit sensor readings
     */
    void sendSensorData() {
        sendDoc.clear();
        sendDoc["type"] = "sensor_data";
        sendDoc["timestamp"] = millis();     // Include current timestamp
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Transmit ultrasonic sensor data
     * @param front, right, back, left: Distance readings in cm from each sensor
     */
    void sendUltrasonicData(float front, float right, float back, float left) {
        sendDoc.clear();
        sendDoc["type"] = "ultrasonic";
        sendDoc["front"] = front;            // Front sensor distance
        sendDoc["right"] = right;            // Right sensor distance
        sendDoc["back"] = back;              // Back sensor distance
        sendDoc["left"] = left;              // Left sensor distance
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Transmit current motor status
     * @param status: Motor status string ("stopped", "moving", "avoiding", etc.)
     */
    void sendMotorStatus(String status) {
        sendDoc.clear();
        sendDoc["type"] = "motor_status";
        sendDoc["status"] = status;
        sendDoc["timestamp"] = millis();     // Include timestamp for tracking
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Transmit obstacle avoidance direction
     * @param direction: Avoidance direction ("forward", "backward", "left", "right", "stop")
     */
    void sendAvoidanceDirection(String direction) {
        sendDoc.clear();
        sendDoc["type"] = "avoidance";
        sendDoc["direction"] = direction;    // Recommended avoidance direction
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    /**
     * Check if system is in safe mode
     * @return: true if in safe mode (communication timeout), false otherwise
     */
    bool isInSafeMode() {
        return safeMode;
    }

    /**
     * Get the current operating mode
     * @return: "auto" or "manual"
     */
    String getOperatingMode() {
        return operatingMode;
    }

    /**
     * Reset safe mode and communication timing
     * Use to manually clear safe mode state
     */
    void resetSafeMode() {
        safeMode = false;
        lastReceiveTime = millis();          // Reset communication timer
    }

    /**
     * Configure data transmission interval
     * @param interval: Time between data transmissions in milliseconds (10-5000ms)
     */
    void setSendInterval(unsigned long interval) {
        // Validate interval range (10ms to 5 seconds)
        if (interval >= 10 && interval <= 5000) {
            sendInterval = interval;
        }
    }

    /**
     * Get current data transmission interval
     * @return: Current transmission interval in milliseconds
     */
    unsigned long getSendInterval() {
        return sendInterval;
    }

    /**
     * Check if new motor command has been received
     * @return: true if new motor command is available
     */
    bool hasMotorCommand() {
        return hasNewMotorCommand;
    }

    /**
     * Get the last received motor direction and clear command flag
     * @return: Motor direction string
     */
    String getMotorDirection() {
        hasNewMotorCommand = false;  // Clear flag after reading
        return lastMotorDirection;
    }

    /**
     * Get the last received motor speed
     * @return: Motor speed value (0-255)
     */
    int getMotorSpeed() {
        return lastMotorSpeed;
    }
};

#endif
