#ifndef ALERT_H
#define ALERT_H

class AlertController {
private:
    int redLedPin;
    int buzzerPin;
    unsigned long lastBlinkTime;
    bool ledState;
    bool isAlerting;

public:
    AlertController(int redPin, int buzzer) {
        redLedPin = redPin;
        buzzerPin = buzzer;
        lastBlinkTime = 0;
        ledState = false;
        isAlerting = false;

        pinMode(redLedPin, OUTPUT);
        pinMode(buzzerPin, OUTPUT);

        digitalWrite(redLedPin, LOW);
        digitalWrite(buzzerPin, LOW);
    }

    void update() {
        if (isAlerting) {
            if (millis() - lastBlinkTime >= 500) {
                ledState = !ledState;
                digitalWrite(redLedPin, ledState);
                lastBlinkTime = millis();
            }
        }
    }

    void startAlert() {
        isAlerting = true;
        digitalWrite(buzzerPin, HIGH);
    }

    void stopAlert() {
        isAlerting = false;
        digitalWrite(redLedPin, LOW);
        digitalWrite(buzzerPin, LOW);
        ledState = false;
    }

    void buzzerOn() {
        digitalWrite(buzzerPin, HIGH);
    }

    void buzzerOff() {
        digitalWrite(buzzerPin, LOW);
    }

    void redLedBlink() {
        startAlert();
    }

    void allOff() {
        stopAlert();
    }
};

#endif
