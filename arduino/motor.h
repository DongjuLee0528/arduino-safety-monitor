/*
 * motor.h – MotorController for L298N Dual H-Bridge
 *
 * Responsibility:
 *   Drives four TT DC motors through an L298N motor driver.
 *   Provides a clean movement API; all pin numbers and speed values
 *   are kept inside this class or in config.h.
 *
 * Design notes:
 *   - Constructor stores pin numbers only.  No GPIO writes occur there,
 *     because Arduino global constructors run before the hardware is ready.
 *   - Call begin() inside setup() to configure pins and enter a known stop.
 *   - setSpeed() clamps its argument to [MOTOR_SPEED_MIN, MOTOR_SPEED_MAX]
 *     and stores the result; subsequent movement calls use the stored speed.
 *   - All analogWrite() calls use the internally stored, clamped speed value.
 *   - left() and right() are aliases for turnLeft() and turnRight() to satisfy
 *     the required API surface.
 *   - Motor direction inversion can be accommodated by negating the per-side
 *     HIGH/LOW values in forward()/backward() without restructuring the class.
 *
 * Public API:
 *   begin()       – configure GPIO, enter stopped state
 *   forward()     – both motors forward at stored speed
 *   backward()    – both motors backward at stored speed
 *   left()        – counter-clockwise point turn at stored speed
 *   right()       – clockwise point turn at stored speed
 *   stop()        – brake (both H-bridge sides HIGH, PWM = 0)
 *   setSpeed(v)   – clamp v and store as current speed
 *   getSpeed()    – return current stored speed
 */

#ifndef MOTOR_H
#define MOTOR_H

#include "config.h"

class MotorController {
private:
    int _lf, _lb, _rf, _rb;  // Direction pins
    int _lpwm, _rpwm;         // PWM pins
    int _speed;               // Current clamped speed

    int clamp(int v) const {
        if (v < MOTOR_SPEED_MIN) return MOTOR_SPEED_MIN;
        if (v > MOTOR_SPEED_MAX) return MOTOR_SPEED_MAX;
        return v;
    }

    void applyPWM() {
        analogWrite(_lpwm, _speed);
        analogWrite(_rpwm, _speed);
    }

public:
    /*
     * Constructor – stores pin numbers only.
     * Does NOT call pinMode() or digitalWrite().
     */
    MotorController(int lf, int lb, int rf, int rb, int lpwm, int rpwm)
        : _lf(lf), _lb(lb), _rf(rf), _rb(rb),
          _lpwm(lpwm), _rpwm(rpwm),
          _speed(MOTOR_SPEED_DEFAULT) {}

    /*
     * begin() – call once from setup().
     * Configures all pins as OUTPUT and enters a known stopped state.
     */
    void begin() {
        pinMode(_lf,   OUTPUT);
        pinMode(_lb,   OUTPUT);
        pinMode(_rf,   OUTPUT);
        pinMode(_rb,   OUTPUT);
        pinMode(_lpwm, OUTPUT);
        pinMode(_rpwm, OUTPUT);
        stop();
    }

    /*
     * setSpeed() – clamp v to [MOTOR_SPEED_MIN, MOTOR_SPEED_MAX] and store.
     * Does not change current direction.
     */
    void setSpeed(int v) {
        _speed = clamp(v);
    }

    int getSpeed() const {
        return _speed;
    }

    void forward() {
        digitalWrite(_lf,  HIGH);
        digitalWrite(_lb,  LOW);
        digitalWrite(_rf,  HIGH);
        digitalWrite(_rb,  LOW);
        applyPWM();
    }

    void backward() {
        digitalWrite(_lf,  LOW);
        digitalWrite(_lb,  HIGH);
        digitalWrite(_rf,  LOW);
        digitalWrite(_rb,  HIGH);
        applyPWM();
    }

    /*
     * turnLeft() – counter-clockwise point turn.
     * Left motor reverses, right motor drives forward.
     */
    void turnLeft() {
        digitalWrite(_lf,  LOW);
        digitalWrite(_lb,  HIGH);
        digitalWrite(_rf,  HIGH);
        digitalWrite(_rb,  LOW);
        applyPWM();
    }

    /*
     * turnRight() – clockwise point turn.
     * Left motor drives forward, right motor reverses.
     */
    void turnRight() {
        digitalWrite(_lf,  HIGH);
        digitalWrite(_lb,  LOW);
        digitalWrite(_rf,  LOW);
        digitalWrite(_rb,  HIGH);
        applyPWM();
    }

    void left()  { turnLeft();  }
    void right() { turnRight(); }

    /*
     * stop() – brake mode: both H-bridge sides HIGH, PWM zero.
     * Safe to call before begin() (pins may not be configured yet in that case,
     * but the method is idempotent after begin()).
     */
    void stop() {
        digitalWrite(_lf,  HIGH);
        digitalWrite(_lb,  HIGH);
        digitalWrite(_rf,  HIGH);
        digitalWrite(_rb,  HIGH);
        analogWrite(_lpwm, MOTOR_SPEED_MIN);
        analogWrite(_rpwm, MOTOR_SPEED_MIN);
    }
};

#endif
