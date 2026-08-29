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
 *   distanceAvailable(n)  – true when _valid[n] is set AND the reading is fresh
 *                            (_valid is cleared immediately on timeout/invalid pulse;
 *                             freshness = within SENSOR_STALE_TIMEOUT_MS as a second guard)
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
    int frontTrig, frontEcho; // Front HC-SR04 trigger and echo pins
    int backTrig,  backEcho;  // Rear HC-SR04 trigger and echo pins
    int leftTrig,  leftEcho;  // Left HC-SR04 trigger and echo pins
    int rightTrig, rightEcho; // Right HC-SR04 trigger and echo pins

    unsigned long lastMeasurement; // millis() timestamp of the last measureSensor() call
    int           currentSensor;   // Index of the next sensor to read (0-3, round-robin)

    float         measurements[4][ULTRASONIC_SAMPLES]; // Circular buffer of raw distance readings per sensor
    int           measureCount[4];                     // Total number of readings ever stored per sensor
    unsigned long _lastValidTime[4];                   // millis() of the most recent valid echo per sensor
    bool          _valid[4];                           // True when the last pulse was valid (not a timeout)

    void measureSensor(int sensor) {
        int trigPin, echoPin;
        // Select trigger/echo pins for the requested sensor index
        switch (sensor) {
            case 0: trigPin = frontTrig; echoPin = frontEcho; break;
            case 1: trigPin = rightTrig; echoPin = rightEcho; break;
            case 2: trigPin = backTrig;  echoPin = backEcho;  break;
            case 3: trigPin = leftTrig;  echoPin = leftEcho;  break;
            default: return; // Ignore out-of-range indices
        }

        // Generate a 10 us HIGH pulse to trigger the HC-SR04 measurement
        digitalWrite(trigPin, LOW);
        delayMicroseconds(2);           // Ensure trigger line is clean before pulsing
        digitalWrite(trigPin, HIGH);
        delayMicroseconds(10);          // 10 us pulse as required by HC-SR04 datasheet
        digitalWrite(trigPin, LOW);

        // Measure the echo pulse width; returns 0 on timeout
        long duration = pulseIn(echoPin, HIGH, ULTRASONIC_TIMEOUT_US);
        // Convert to cm: speed of sound 0.034 cm/us, divide by 2 for one-way trip
        float distance = duration * 0.034f / 2.0f;

        if (distance > 0 && distance < MAX_SENSOR_RANGE_CM) {
            // Store in circular buffer and mark this reading as valid
            measurements[sensor][measureCount[sensor] % ULTRASONIC_SAMPLES] = distance;
            measureCount[sensor]++;
            _lastValidTime[sensor] = millis();
            _valid[sensor]         = true;
        } else {
            _valid[sensor] = false; // Timeout or out-of-range pulse
        }
    }

public:
    /*
     * Constructor – stores pin numbers and resets state only.
     * Does NOT call pinMode() or any hardware access.
     * Call begin() inside setup() to configure GPIO.
     */
    UltrasonicSensor(int ft, int fe, int bt, int be,
                     int lt, int le, int rt, int re) {
        frontTrig = ft; frontEcho = fe;
        backTrig  = bt; backEcho  = be;
        leftTrig  = lt; leftEcho  = le;
        rightTrig = rt; rightEcho = re;

        lastMeasurement = 0;
        currentSensor   = 0;

        for (int i = 0; i < 4; i++) {
            measureCount[i]   = 0;
            _lastValidTime[i] = 0;
            _valid[i]         = false;
            for (int j = 0; j < ULTRASONIC_SAMPLES; j++) {
                measurements[i][j] = 0;
            }
        }
    }

    /*
     * begin() – configure GPIO pin modes.
     * Call once from setup() before the main control loop starts.
     */
    void begin() {
        pinMode(frontTrig, OUTPUT); digitalWrite(frontTrig, LOW); pinMode(frontEcho, INPUT);
        pinMode(backTrig,  OUTPUT); digitalWrite(backTrig,  LOW); pinMode(backEcho,  INPUT);
        pinMode(leftTrig,  OUTPUT); digitalWrite(leftTrig,  LOW); pinMode(leftEcho,  INPUT);
        pinMode(rightTrig, OUTPUT); digitalWrite(rightTrig, LOW); pinMode(rightEcho, INPUT);
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

    void measureAllOnce() {
        // Diagnostic path: read all sensors immediately instead of waiting for round-robin scheduling.
        measureSensor(0);
        measureSensor(1);
        measureSensor(2);
        measureSensor(3);
        currentSensor   = 0;
        lastMeasurement = millis();
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
     * distanceAvailable() – returns true when:
     *   1. measureCount > 0  (at least one successful reading ever)
     *   2. _valid[sensor]    (most recent attempt was a valid pulse, not a timeout)
     *   3. timestamp fresh   (within SENSOR_STALE_TIMEOUT_MS as a secondary guard)
     * _valid is set false immediately when pulseIn() returns 0 (timeout or no echo),
     * so a disconnected sensor becomes unavailable on the very next measurement attempt.
     */
    bool distanceAvailable(int sensor) const {
        if (measureCount[sensor] == 0) return false;
        if (!_valid[sensor])           return false;
        return (millis() - _lastValidTime[sensor]) <= SENSOR_STALE_TIMEOUT_MS;
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
     * This legacy helper ignores distanceAvailable(); callers needing fail-safe
     * unknown-sensor handling should use NavigationManager.
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
