#ifndef BRIDGE_RPC_H
#define BRIDGE_RPC_H

#include <ArduinoJson.h>
#include "motor.h"
#include "alert.h"

class BridgeRPC {
private:
    unsigned long lastReceiveTime;
    unsigned long lastSendTime;
    String inputBuffer;
    StaticJsonDocument<256> receiveDoc;
    StaticJsonDocument<256> sendDoc;
    bool safeMode;
    MotorController* motorController;
    AlertController* alertController;

    const unsigned long TIMEOUT_THRESHOLD = 500;
    unsigned long sendInterval;

public:
    BridgeRPC(MotorController* motor, AlertController* alert) {
        lastReceiveTime = 0;
        lastSendTime = 0;
        safeMode = false;
        inputBuffer = "";
        motorController = motor;
        alertController = alert;
        sendInterval = 100;
    }

    void update() {
        checkTimeout();
        receiveCommands();
        if (millis() - lastSendTime >= sendInterval) {
            sendSensorData();
            lastSendTime = millis();
        }
    }

    void checkTimeout() {
        if (millis() - lastReceiveTime > TIMEOUT_THRESHOLD && lastReceiveTime > 0) {
            if (!safeMode) {
                safeMode = true;
                triggerSafeMode();
            }
        }
    }

    void triggerSafeMode() {
        sendDoc.clear();
        sendDoc["status"] = "timeout";
        sendDoc["action"] = "emergency_stop";
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    void receiveCommands() {
        while (Serial.available()) {
            char c = Serial.read();
            if (c == '\n') {
                processCommand(inputBuffer);
                inputBuffer = "";
            } else {
                inputBuffer += c;
            }
        }
    }

    void processCommand(String command) {
        deserializeJson(receiveDoc, command);

        if (receiveDoc.containsKey("cmd")) {
            lastReceiveTime = millis();
            safeMode = false;

            String cmd = receiveDoc["cmd"];

            if (cmd == "motor") {
                String direction = receiveDoc["direction"];
                int speed = receiveDoc["speed"];
                handleMotorCommand(direction, speed);
            }
            else if (cmd == "led") {
                String color = receiveDoc["value"];
                handleLedCommand(color);
            }
            else if (cmd == "buzzer") {
                String state = receiveDoc["value"];
                handleBuzzerCommand(state);
            }
            else if (cmd == "ping") {
                sendPongResponse();
            }
        }
    }

    void handleMotorCommand(String direction, int speed) {
        if (motorController) {
            motorController->processCommand(direction, speed);
        }
        sendDoc.clear();
        sendDoc["type"] = "motor_ack";
        sendDoc["direction"] = direction;
        sendDoc["speed"] = speed;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    void handleLedCommand(String color) {
        if (alertController) {
            if (color == "red") {
                alertController->startAlert();
            } else if (color == "off") {
                alertController->stopAlert();
            }
        }
        sendDoc.clear();
        sendDoc["type"] = "led_ack";
        sendDoc["color"] = color;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    void handleBuzzerCommand(String state) {
        if (alertController) {
            if (state == "on") {
                alertController->buzzerOn();
            } else if (state == "off") {
                alertController->buzzerOff();
            }
        }
        sendDoc.clear();
        sendDoc["type"] = "buzzer_ack";
        sendDoc["state"] = state;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    void sendPongResponse() {
        sendDoc.clear();
        sendDoc["type"] = "pong";
        sendDoc["timestamp"] = millis();
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    void sendSensorData() {
        sendDoc.clear();
        sendDoc["type"] = "sensor_data";
        sendDoc["timestamp"] = millis();
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    void sendUltrasonicData(float front, float right, float back, float left) {
        sendDoc.clear();
        sendDoc["type"] = "ultrasonic";
        sendDoc["front"] = front;
        sendDoc["right"] = right;
        sendDoc["back"] = back;
        sendDoc["left"] = left;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    void sendMotorStatus(String status) {
        sendDoc.clear();
        sendDoc["type"] = "motor_status";
        sendDoc["status"] = status;
        sendDoc["timestamp"] = millis();
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    void sendAvoidanceDirection(String direction) {
        sendDoc.clear();
        sendDoc["type"] = "avoidance";
        sendDoc["direction"] = direction;
        String output;
        serializeJson(sendDoc, output);
        Serial.println(output);
    }

    bool isInSafeMode() {
        return safeMode;
    }

    void resetSafeMode() {
        safeMode = false;
        lastReceiveTime = millis();
    }

    void setSendInterval(unsigned long interval) {
        if (interval >= 10 && interval <= 5000) {
            sendInterval = interval;
        }
    }

    unsigned long getSendInterval() {
        return sendInterval;
    }
};

#endif
