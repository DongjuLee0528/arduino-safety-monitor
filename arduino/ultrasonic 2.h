#ifndef ULTRASONIC_H
#define ULTRASONIC_H

class UltrasonicSensor {
private:
    int frontTrig, frontEcho;
    int backTrig, backEcho;
    int leftTrig, leftEcho;
    int rightTrig, rightEcho;

    unsigned long lastMeasurement;
    int currentSensor;
    float measurements[4][3];
    int measureCount[4];

    const float THRESHOLD = 30.0;
    const int MEASURE_INTERVAL = 50;
    const int SAMPLES = 3;

public:
    UltrasonicSensor(int ft, int fe, int bt, int be, int lt, int le, int rt, int re) {
        frontTrig = ft; frontEcho = fe;
        backTrig = bt; backEcho = be;
        leftTrig = lt; leftEcho = le;
        rightTrig = rt; rightEcho = re;

        pinMode(frontTrig, OUTPUT); pinMode(frontEcho, INPUT);
        pinMode(backTrig, OUTPUT); pinMode(backEcho, INPUT);
        pinMode(leftTrig, OUTPUT); pinMode(leftEcho, INPUT);
        pinMode(rightTrig, OUTPUT); pinMode(rightEcho, INPUT);

        lastMeasurement = 0;
        currentSensor = 0;

        for(int i = 0; i < 4; i++) {
            measureCount[i] = 0;
            for(int j = 0; j < SAMPLES; j++) {
                measurements[i][j] = 0;
            }
        }
    }

    void update() {
        if(millis() - lastMeasurement >= MEASURE_INTERVAL) {
            measureSensor(currentSensor);
            currentSensor = (currentSensor + 1) % 4;
            lastMeasurement = millis();
        }
    }

    void measureSensor(int sensor) {
        int trigPin, echoPin;

        switch(sensor) {
            case 0: trigPin = frontTrig; echoPin = frontEcho; break;
            case 1: trigPin = rightTrig; echoPin = rightEcho; break;
            case 2: trigPin = backTrig; echoPin = backEcho; break;
            case 3: trigPin = leftTrig; echoPin = leftEcho; break;
        }

        digitalWrite(trigPin, LOW);
        delayMicroseconds(2);
        digitalWrite(trigPin, HIGH);
        delayMicroseconds(10);
        digitalWrite(trigPin, LOW);

        long duration = pulseIn(echoPin, HIGH, 30000);
        float distance = duration * 0.034 / 2;

        if(distance > 0 && distance < 300) {
            measurements[sensor][measureCount[sensor] % SAMPLES] = distance;
            measureCount[sensor]++;
        }
    }

    float getAverageDistance(int sensor) {
        if(measureCount[sensor] == 0) return 300;

        int count = min(measureCount[sensor], SAMPLES);
        float sum = 0;

        for(int i = 0; i < count; i++) {
            sum += measurements[sensor][i];
        }

        return sum / count;
    }

    bool hasObstacle(int sensor) {
        return getAverageDistance(sensor) < THRESHOLD;
    }

    String getAvoidanceDirection() {
        bool front = hasObstacle(0);
        bool right = hasObstacle(1);
        bool back = hasObstacle(2);
        bool left = hasObstacle(3);

        if(!front && !left && !right) return "forward";
        if(!back && !left && !right) return "backward";
        if(!left && !front && !back) return "left";
        if(!right && !front && !back) return "right";

        if(!front) return "forward";
        if(!back) return "backward";
        if(!left) return "left";
        if(!right) return "right";

        return "stop";
    }

    float getFrontDistance() { return getAverageDistance(0); }
    float getRightDistance() { return getAverageDistance(1); }
    float getBackDistance() { return getAverageDistance(2); }
    float getLeftDistance() { return getAverageDistance(3); }
};

#endif
