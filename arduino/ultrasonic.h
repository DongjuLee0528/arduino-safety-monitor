/*
 * Ultrasonic Sensor Library for HC-SR04 Obstacle Detection
 *
 * This library manages four HC-SR04 ultrasonic sensors for 360-degree obstacle detection.
 * Features:
 * - Non-blocking sensor readings with round-robin scheduling
 * - Multi-sample averaging for noise reduction
 * - Configurable obstacle detection threshold
 * - Intelligent avoidance direction recommendation
 *
 * Sensor Layout:
 * 0: Front sensor - primary direction
 * 1: Right sensor - right side detection
 * 2: Back sensor - reverse direction
 * 3: Left sensor - left side detection
 */

#ifndef ULTRASONIC_H
#define ULTRASONIC_H

#include "config.h"

/**
 * UltrasonicSensor Class
 *
 * Manages multiple HC-SR04 sensors for autonomous obstacle avoidance.
 * Uses time-multiplexed readings to avoid interference between sensors.
 */
class UltrasonicSensor {
private:
    // Pin assignments for each HC-SR04 sensor
    int frontTrig, frontEcho;   // Front-facing sensor pins
    int backTrig, backEcho;     // Rear-facing sensor pins
    int leftTrig, leftEcho;     // Left-facing sensor pins
    int rightTrig, rightEcho;   // Right-facing sensor pins

    // Timing and measurement state
    unsigned long lastMeasurement; // Timestamp of last sensor reading
    int currentSensor;             // Index of sensor currently being read (0-3)

    // Multi-sample measurement arrays for noise filtering
    float measurements[4][3];      // Store last 3 readings for each sensor
    int measureCount[4];           // Number of valid readings per sensor

    // Configuration constants
    const float THRESHOLD = 30.0;      // Obstacle detection distance in cm
    const int MEASURE_INTERVAL = 50;   // Time between readings in milliseconds
    const int SAMPLES = 3;             // Number of samples to average

public:
    /**
     * Constructor - Initialize ultrasonic sensor array
     * @param ft, fe: Front sensor trigger and echo pins
     * @param bt, be: Back sensor trigger and echo pins
     * @param lt, le: Left sensor trigger and echo pins
     * @param rt, re: Right sensor trigger and echo pins
     */
    UltrasonicSensor(int ft, int fe, int bt, int be, int lt, int le, int rt, int re) {
        // Store pin assignments
        frontTrig = ft; frontEcho = fe;
        backTrig = bt; backEcho = be;
        leftTrig = lt; leftEcho = le;
        rightTrig = rt; rightEcho = re;

        // Configure pins - trigger pins as outputs, echo pins as inputs
        // TODO A10: frontEcho를 입력으로 설정하세요 (신호를 읽으려면 INPUT)
        pinMode(frontTrig, OUTPUT); pinMode(frontEcho, __);
        pinMode(backTrig, OUTPUT); pinMode(backEcho, INPUT);
        pinMode(leftTrig, OUTPUT); pinMode(leftEcho, INPUT);
        pinMode(rightTrig, OUTPUT); pinMode(rightEcho, INPUT);

        // Initialize timing variables
        lastMeasurement = 0;
        currentSensor = 0;

        // Initialize measurement arrays
        for(int i = 0; i < 4; i++) {
            measureCount[i] = 0;
            for(int j = 0; j < SAMPLES; j++) {
                measurements[i][j] = 0;
            }
        }
    }

    /**
     * Update method - call regularly in main loop
     * Performs non-blocking sensor readings using time-multiplexing
     * to avoid interference between sensors
     */
    void update() {
        if(millis() - lastMeasurement >= MEASURE_INTERVAL) {
            measureSensor(currentSensor);                   // Read current sensor
            currentSensor = (currentSensor + 1) % 4;       // Advance to next sensor
            lastMeasurement = millis();                     // Update timestamp
        }
    }

    /**
     * Measure distance from a specific sensor
     * @param sensor: Sensor index (0=front, 1=right, 2=back, 3=left)
     */
    void measureSensor(int sensor) {
        int trigPin, echoPin;

        // Select pins for the specified sensor
        switch(sensor) {
            case 0: trigPin = frontTrig; echoPin = frontEcho; break;
            case 1: trigPin = rightTrig; echoPin = rightEcho; break;
            case 2: trigPin = backTrig; echoPin = backEcho; break;
            case 3: trigPin = leftTrig; echoPin = leftEcho; break;
        }

        // Generate ultrasonic pulse: LOW -> HIGH -> LOW
        digitalWrite(trigPin, LOW);
        delayMicroseconds(2);         // Ensure clean LOW state
        // TODO A11: 초음파를 발사하려면 trigPin을 일시적으로 HIGH로 설정하세요 (digitalWrite HIGH/LOW)
        digitalWrite(trigPin, __);
        delayMicroseconds(10);        // 10μs pulse duration
        digitalWrite(trigPin, LOW);

        // Measure echo pulse duration (timeout configured in config.h)
        long duration = pulseIn(echoPin, HIGH, ULTRASONIC_TIMEOUT_US);

        // Convert duration to distance: speed of sound = 340m/s = 0.034cm/μs
        // Divide by 2 because sound travels to object and back
        float distance = duration * 0.034 / 2;

        // Store valid measurements (filter out invalid readings)
        if(distance > 0 && distance < 300) {
            measurements[sensor][measureCount[sensor] % SAMPLES] = distance;
            measureCount[sensor]++;
        }
    }

    /**
     * Calculate average distance for noise reduction
     * @param sensor: Sensor index (0-3)
     * @return: Average distance in cm, or 300cm if no valid readings
     */
    float getAverageDistance(int sensor) {
        if(measureCount[sensor] == 0) return 300; // Return max range if no readings

        int count = min(measureCount[sensor], SAMPLES); // Use available samples
        float sum = 0;

        // Sum all available measurements
        for(int i = 0; i < count; i++) {
            // TODO A12: 측정값을 sum에 누적하세요 (+= 연산자)
            sum += __;
        }

        return sum / count; // Return average
    }

    /**
     * Check if sensor detects an obstacle within threshold
     * @param sensor: Sensor index (0-3)
     * @return: true if obstacle detected, false otherwise
     */
    bool hasObstacle(int sensor) {
        // TODO A13: 평균 거리가 THRESHOLD보다 작으면 true를 반환하세요 (비교 연산자)
        return getAverageDistance(sensor) < __;
    }

    /**
     * Determine best avoidance direction based on all sensor readings
     * Uses priority system: prefer forward movement, then consider all options
     * @return: Direction string ("forward", "backward", "left", "right", or "stop")
     */
    String getAvoidanceDirection() {
        // Get obstacle status for each direction
        bool front = hasObstacle(0);
        bool right = hasObstacle(1);
        bool back = hasObstacle(2);
        bool left = hasObstacle(3);

        // Priority 1: Preferred directions if multiple paths are clear
        if(!front && !left && !right) return "forward";    // Forward preferred
        if(!back && !left && !right) return "backward";    // Reverse if front blocked
        if(!left && !front && !back) return "left";        // Left turn
        if(!right && !front && !back) return "right";      // Right turn

        // Priority 2: Single clear directions
        if(!front) return "forward";
        if(!back) return "backward";
        if(!left) return "left";
        if(!right) return "right";

        // All directions blocked - stop and alert
        // TODO A14: 모든 방향이 막혔을 때 "stop"을 반환하세요 (return 문)
        return __;
    }

    // Convenience methods for accessing individual sensor readings
    float getFrontDistance() { return getAverageDistance(0); }  // Front sensor
    float getRightDistance() { return getAverageDistance(1); }  // Right sensor
    float getBackDistance() { return getAverageDistance(2); }   // Back sensor
    float getLeftDistance() { return getAverageDistance(3); }   // Left sensor
};

#endif
