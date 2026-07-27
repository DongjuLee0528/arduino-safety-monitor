/*
 * UltrasonicSensor
 *
 * Responsibility:
 *   Manages four HC-SR04 sensors (front, rear, left, right).
 *   Performs non-blocking, time-multiplexed readings and provides
 *   rolling-average distances for noise reduction.
 *
 * Sensor index mapping (used internally):
 *   0 – front
 *   1 – right
 *   2 – back (rear)
 *   3 – left
 *
 * Public API:
 *   update()              – advance the round-robin measurement scheduler
 *   getFrontDistance()    – averaged front distance in cm
 *   getBackDistance()     – averaged rear distance in cm
 *   getLeftDistance()     – averaged left distance in cm
 *   getRightDistance()    – averaged right distance in cm
 *   distanceAvailable(n)  – true once at least one valid reading exists for sensor n
 *   hasObstacle(n)        – true when averaged distance < OBSTACLE_THRESHOLD_CM
 *   getAvoidanceDirection() – heuristic "safest direction" string for legacy callers
 *
 * Configuration constants are defined in config.h:
 *   ULTRASONIC_TIMEOUT_US   – pulseIn() hard timeout
 *   ULTRASONIC_SAMPLES      – rolling-average window
 *   ULTRASONIC_INTERVAL_MS  – minimum ms between readings
 *   OBSTACLE_THRESHOLD_CM   – obstacle detection threshold
 *   MAX_SENSOR_RANGE_CM     – value returned when no valid reading exists
 */

#ifndef ULTRASONIC_H
#define ULTRASONIC_H

#include "config.h"

class UltrasonicSensor {
private:
    int frontTrig, frontEcho;
    int backTrig,  backEcho;
    int leftTrig,  leftEcho;
    int rightTrig, rightEcho;

    unsigned long lastMeasurement;
    int           currentSensor;

    float measurements[4][ULTRASONIC_SAMPLES];
    int   measureCount[4];

    void measureSensor(int sensor) {
        int trigPin, echoPin;
        switch (sensor) {
            case 0: trigPin = frontTrig; echoPin = frontEcho; break;
            case 1: trigPin = rightTrig; echoPin = rightEcho; break;
            case 2: trigPin = backTrig;  echoPin = backEcho;  break;
            case 3: trigPin = leftTrig;  echoPin = leftEcho;  break;
            default: return;
        }

        digitalWrite(trigPin, LOW);
        delayMicroseconds(2);
        digitalWrite(trigPin, HIGH);
        delayMicroseconds(10);
        digitalWrite(trigPin, LOW);

        long duration = pulseIn(echoPin, HIGH, ULTRASONIC_TIMEOUT_US);
        float distance = duration * 0.034f / 2.0f;

        if (distance > 0 && distance < MAX_SENSOR_RANGE_CM) {
            measurements[sensor][measureCount[sensor] % ULTRASONIC_SAMPLES] = distance;
            measureCount[sensor]++;
        }
    }

public:
    UltrasonicSensor(int ft, int fe, int bt, int be,
                     int lt, int le, int rt, int re) {
        frontTrig = ft; frontEcho = fe;
        backTrig  = bt; backEcho  = be;
        leftTrig  = lt; leftEcho  = le;
        rightTrig = rt; rightEcho = re;

        pinMode(frontTrig, OUTPUT); pinMode(frontEcho, INPUT);
        pinMode(backTrig,  OUTPUT); pinMode(backEcho,  INPUT);
        pinMode(leftTrig,  OUTPUT); pinMode(leftEcho,  INPUT);
        pinMode(rightTrig, OUTPUT); pinMode(rightEcho, INPUT);

        lastMeasurement = 0;
        currentSensor   = 0;

        for (int i = 0; i < 4; i++) {
            measureCount[i] = 0;
            for (int j = 0; j < ULTRASONIC_SAMPLES; j++) {
                measurements[i][j] = 0;
            }
        }
    }

    /*
     * update() – advance the non-blocking round-robin scheduler.
     * Call once per main loop iteration.
     */
    void update() {
        if (millis() - lastMeasurement >= ULTRASONIC_INTERVAL_MS) {
            measureSensor(currentSensor);
            currentSensor   = (currentSensor + 1) % 4;
            lastMeasurement = millis();
        }
    }

    /*
     * getAverageDistance() – rolling average of the last ULTRASONIC_SAMPLES
     * readings for sensor `sensor`.
     * Returns MAX_SENSOR_RANGE_CM when no valid reading has been received.
     */
    float getAverageDistance(int sensor) const {
        if (measureCount[sensor] == 0) return MAX_SENSOR_RANGE_CM;

        int   count = min(measureCount[sensor], ULTRASONIC_SAMPLES);
        float sum   = 0;
        for (int i = 0; i < count; i++) {
            sum += measurements[sensor][i];
        }
        return sum / count;
    }

    /*
     * distanceAvailable() – returns true once at least one valid measurement
     * exists for the given sensor index (0-3).
     * Use this before trusting getAverageDistance() at boot time.
     */
    bool distanceAvailable(int sensor) const {
        return measureCount[sensor] > 0;
    }

    /*
     * hasObstacle() – returns true when the averaged distance for sensor
     * `sensor` is below OBSTACLE_THRESHOLD_CM.
     */
    bool hasObstacle(int sensor) const {
        return getAverageDistance(sensor) < OBSTACLE_THRESHOLD_CM;
    }

    float getFrontDistance() const { return getAverageDistance(0); }
    float getRightDistance() const { return getAverageDistance(1); }
    float getBackDistance()  const { return getAverageDistance(2); }
    float getLeftDistance()  const { return getAverageDistance(3); }

    /*
     * getAvoidanceDirection() – heuristic best escape direction.
     * Kept for backward compatibility; NavigationManager is preferred.
     * Returns "forward" | "backward" | "left" | "right" | "stop"
     */
    String getAvoidanceDirection() const {
        bool front = hasObstacle(0);
        bool right = hasObstacle(1);
        bool back  = hasObstacle(2);
        bool left  = hasObstacle(3);

        if (!front && !left && !right) return "forward";
        if (!back  && !left && !right) return "backward";
        if (!left  && !front && !back) return "left";
        if (!right && !front && !back) return "right";

        if (!front) return "forward";
        if (!back)  return "backward";
        if (!left)  return "left";
        if (!right) return "right";

        return "stop";
    }
};

#endif
