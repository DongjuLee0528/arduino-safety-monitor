/*
 * Alert System Library for Robot Status Indication
 *
 * This library manages visual and audible alerts for the robot system including:
 * - LED status indication with blinking patterns
 * - Buzzer alerts for warnings and system states
 * - Non-blocking operation with automatic timing
 *
 * Alert Types:
 * - Emergency alerts: Continuous buzzer + blinking LED
 * - Status indication: LED patterns
 * - Manual control: Individual component control
 */

#ifndef ALERT_H
#define ALERT_H

#include "config.h"

/**
 * AlertController Class
 *
 * Manages LED and buzzer outputs for system status indication.
 * Provides both automatic alert patterns and manual control.
 */
class AlertController {
private:
    // Hardware pin assignments
    int redLedPin;              // Red LED pin for visual alerts
    int buzzerPin;              // Buzzer pin for audio alerts

    // Timing and state variables for non-blocking operation
    unsigned long lastBlinkTime; // Timestamp of last LED state change
    bool ledState;              // Current LED state (on/off)
    bool isAlerting;            // Flag indicating active alert mode

public:
    /**
     * Constructor - Initialize alert system with pin assignments
     * @param redPin: Digital pin connected to red LED
     * @param buzzer: Digital pin connected to buzzer
     */
    AlertController(int redPin, int buzzer) {
        // Store pin assignments
        redLedPin = redPin;
        buzzerPin = buzzer;

        // Initialize state variables
        lastBlinkTime = 0;
        ledState = false;
        isAlerting = false;

        // Configure pins as outputs
        pinMode(redLedPin, OUTPUT);
        pinMode(buzzerPin, OUTPUT);

        // Ensure both outputs start in OFF state
        // TODO A6: 시작 시 redLedPin을 끄세요 (LOW 상태로 초기화)
        digitalWrite(redLedPin, __);
        digitalWrite(buzzerPin, LOW);
    }

    /**
     * Update method - call regularly in main loop
     * Handles non-blocking LED blinking when alert is active
     * Blinks LED at 1Hz (500ms on, 500ms off) during alerts
     */
    void update() {
        // TODO A7: isAlerting일 때만 LED를 업데이트하세요 (if 조건문)
        if (__) {
            // Check if it's time to toggle LED (configured interval)
            if (millis() - lastBlinkTime >= LED_BLINK_INTERVAL_MS) {
                ledState = !ledState;                    // Toggle LED state
                digitalWrite(redLedPin, ledState);       // Apply new state
                lastBlinkTime = millis();                // Update timestamp
            }
        }
    }

    /**
     * Start emergency alert mode
     * Activates continuous buzzer and begins LED blinking pattern
     * Used for obstacle detection, emergency stops, etc.
     */
    void startAlert() {
        // TODO A8: 알림 모드를 활성화하세요 (bool 변수에 true 할당)
        isAlerting = __;
        digitalWrite(buzzerPin, HIGH);        // Turn on buzzer immediately
        // LED blinking is handled by update() method
    }

    /**
     * Stop all alert activities
     * Turns off buzzer, stops LED blinking, and resets alert state
     */
    void stopAlert() {
        isAlerting = false;                   // Disable alert mode
        digitalWrite(redLedPin, LOW);         // Turn off LED
        // TODO A9: buzzerPin을 LOW로 설정하세요 (digitalWrite 함수)
        digitalWrite(buzzerPin, __);
        ledState = false;                     // Reset LED state
    }

    /**
     * Manual buzzer control - turn buzzer on
     * For direct control without affecting LED or alert state
     */
    void buzzerOn() {
        digitalWrite(buzzerPin, HIGH);
    }

    /**
     * Manual buzzer control - turn buzzer off
     * For direct control without affecting LED or alert state
     */
    void buzzerOff() {
        digitalWrite(buzzerPin, LOW);
    }

    /**
     * Start LED blinking pattern
     * Convenience method - equivalent to startAlert()
     */
    void redLedBlink() {
        startAlert();
    }

    /**
     * Turn off all alert components
     * Convenience method - equivalent to stopAlert()
     */
    void allOff() {
        stopAlert();
    }
};

#endif
