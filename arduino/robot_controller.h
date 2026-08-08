/*
 * robot_controller.h – RobotController
 *
 * Responsibility:
 *   Top-level orchestrator.  Wires motor, ultrasonic, navigation, and
 *   communication modules together.  Drives per-loop behaviour.
 *
 *   Does NOT contain navigation logic (→ NavigationManager).
 *   Does NOT contain communication/parsing logic (→ CommunicationManager).
 *   Does NOT contain any alert, LED, or buzzer logic (removed per scope).
 *
 * Operating modes (owned by CommunicationManager, read and acted on here):
 *   MODE_AUTO   – NavigationManager decides movement each loop.
 *   MODE_MANUAL – Pending movement commands from CommunicationManager are executed.
 *
 * Safety handling – all motor actions originate here:
 *   update() processes events in strict priority order:
 *     1. Serial timeout / disconnect  → stop motors, forceManual(), return
 *     2. MOVE_STOP pending           → stop motors, return
 *     3. Manual movement command     → execute via MotorController
 *     4. Autonomous navigation       → NavigationManager → MotorController
 *
 * Serial timeout / disconnect:
 *   Detected via CommunicationManager::isConnected() (set false by
 *   CommunicationManager after SERIAL_CMD_TIMEOUT_MS with no bytes received).
 *   RobotController stops motors, calls resetToManualSafeState() to reset mode
 *   and clear pending state, then returns.
 *   The robot does not move again until isConnected() becomes true and
 *   a new explicit command is received.  This is enforced every loop tick.
 *
 * STOP command:
 *   CommunicationManager sets pendingMove = MOVE_STOP and mode = MANUAL.
 *   RobotController sees MODE_MANUAL, calls hasPendingMove(), gets MOVE_STOP,
 *   stops motors immediately, and returns without further movement.
 *
 * MANUAL command (while in AUTO):
 *   CommunicationManager sets pendingMove = MOVE_STOP and mode = MANUAL.
 *   Same path as STOP above.
 *
 * AUTO command:
 *   CommunicationManager sets mode = AUTO, clears pending.
 *   RobotController resumes autonomous navigation on the next iteration.
 *   Autonomous navigation does not restart automatically after a disconnect;
 *   it requires an explicit AUTO command after reconnection.
 *
 * Fail-safe (sensor availability):
 *   Sensor availability is checked via distanceAvailable() before calling
 *   NavigationManager::update().  Availability flags are passed to
 *   NavigationManager so it can treat missing readings as blocked.
 *
 * Module interaction:
 *
 *   CommunicationManager ──► RobotController ◄── NavigationManager
 *     (mode / pending cmd /    │  (navState)         ▲
 *      isConnected)            │                     │
 *             MotorController ◄┘             UltrasonicSensor
 */

#ifndef ROBOT_CONTROLLER_H
#define ROBOT_CONTROLLER_H

#include "config.h"
#include "motor.h"
#include "ultrasonic.h"
#include "navigation.h"
#include "comm.h"

class RobotController {
private:
    MotorController*      _motor;      // Drives the L298N motor driver pins
    UltrasonicSensor*     _ultrasonic; // Provides averaged distances from four HC-SR04 sensors
    CommunicationManager* _comm;       // Owns serial I/O, mode state, and the motion lease
    NavigationManager     _nav;        // Stateful autonomous navigation decision engine

    unsigned long _lastSendTime; // millis() of the most recent telemetry transmission

    /*
     * runManualMode() – consume the pending command and actuate motors.
     * MOVE_STOP is handled before this is called (see update()).
     * Only non-stop movement commands reach this method.
     */
    void runManualMode() {
        if (_comm->hasPendingMove()) {
            int         spd = _comm->getPendingSpeed();    // Speed paired with this command
            MovementCmd cmd = _comm->consumePendingMove(); // Consume clears the pending slot and lifts _stopLatched
            switch (cmd) {
                case MOVE_FORWARD:  _motor->setSpeed(spd); _motor->forward();  break;
                case MOVE_BACKWARD: _motor->setSpeed(spd); _motor->backward(); break;
                case MOVE_LEFT:     _motor->setSpeed(spd); _motor->left();     break;
                case MOVE_RIGHT:    _motor->setSpeed(spd); _motor->right();    break;
                case MOVE_STOP:     _motor->stop();                            break; // Safety fallback; normally caught earlier
                default:                                                        break;
            }
        }
    }

    void runAutoMode() {
        _ultrasonic->update(); // Advance the round-robin sensor scheduler (one sensor per call)

        // Read availability flags alongside distances; unavailable readings are treated as blocked by NavigationManager
        bool  frontOk = _ultrasonic->distanceAvailable(0);
        float front   = _ultrasonic->getFrontDistance();
        bool  rearOk  = _ultrasonic->distanceAvailable(2);
        float rear    = _ultrasonic->getBackDistance();
        bool  leftOk  = _ultrasonic->distanceAvailable(3);
        float left    = _ultrasonic->getLeftDistance();
        bool  rightOk = _ultrasonic->distanceAvailable(1);
        float right   = _ultrasonic->getRightDistance();

        // Let NavigationManager decide the next movement based on sensor data and availability
        _nav.update(frontOk, front, rearOk, rear, leftOk, left, rightOk, right);
        NavState state = _nav.getState();

        // Translate the navigation decision into a concrete motor command
        switch (state) {
            case NAV_FORWARD:
                _motor->setSpeed(CRUISE_SPEED);
                _motor->forward();
                break;

            case NAV_TURN_LEFT:
                _motor->setSpeed(OBSTACLE_AVOID_SPEED);
                _motor->left();
                break;

            case NAV_TURN_RIGHT:
                _motor->setSpeed(OBSTACLE_AVOID_SPEED);
                _motor->right();
                break;

            case NAV_BACKWARD:
                _motor->setSpeed(OBSTACLE_AVOID_SPEED);
                _motor->backward();
                break;

            case NAV_STOP:
            case NAV_IDLE:
            default:
                _motor->stop(); // All directions blocked or sensor data unavailable
                break;
        }

        // Periodically push sensor distances and navigation state to the Python host
        if (millis() - _lastSendTime >= COMM_SEND_INTERVAL_MS) {
            _comm->sendSensorData(front, rear, left, right);
            _comm->sendStatus(_nav.stateToString().c_str());
            _lastSendTime = millis();
        }
    }

public:
    RobotController(MotorController*      m,
                    UltrasonicSensor*     u,
                    CommunicationManager* c)
        : _motor(m), _ultrasonic(u), _comm(c), _lastSendTime(0) {}

    /*
     * update() – call once per main loop iteration, after comm.update().
     *
     * Safety priority order (highest first):
     *   1. Disconnect / timeout: stop motors, forceManual, return.
     *   2. STOP pending (MODE_MANUAL + MOVE_STOP): stop motors, return.
     *   3. Other manual movement: execute via MotorController.
     *   4. AUTO mode: run NavigationManager → MotorController.
     */
    void update() {
        // Priority 1: serial timeout or host disconnect — stop everything immediately
        if (!_comm->isConnected()) {
            _nav.reset();                    // Discard stale navigation state
            _motor->stop();
            _comm->resetToManualSafeState(); // Clear mode, pending move, and motion lease
            return;
        }

        // Priority 2: helmet-warning active — keep sensors running but hold motors stopped
        _comm->updateWarning();
        if (_comm->isWarningActive()) {
            _nav.reset();
            _ultrasonic->update(); // Keep sensor data fresh during warning window
            _motor->stop();
            return;
        }

        // Priority 3 / 4: normal operation — dispatch to manual or autonomous path
        if (_comm->getMode() == MODE_MANUAL) {
            _nav.reset();      // Reset nav state so it starts fresh if AUTO is re-entered later
            runManualMode();
        } else {
            runAutoMode();     // NavigationManager decides movement based on live sensor readings
        }
    }
};

#endif
