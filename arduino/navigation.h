/*
 * navigation.h – NavigationManager
 *
 * Responsibility:
 *   Owns the autonomous navigation state machine.
 *   Converts validated, available distance readings into a NavState decision.
 *   Does NOT call motor functions.  Does NOT read Serial.
 *
 * Fail-safe policy (Fix 4):
 *   update() requires explicit availability flags alongside each distance value.
 *   If the front reading is unavailable, the result is NAV_STOP regardless of
 *   other sensors.  If a required side or rear reading is unavailable when it
 *   would be needed to choose a direction, that direction is considered blocked.
 *   This ensures missing data is never silently treated as a clear path.
 *
 * State transitions:
 *   NAV_IDLE       – initial / reset state; no data processed yet
 *   NAV_FORWARD    – front clear and available
 *   NAV_TURN_LEFT  – front blocked; right clear+available, left blocked or unavailable
 *   NAV_TURN_RIGHT – front blocked; left clear+available, right blocked or unavailable
 *   NAV_BACKWARD   – front blocked; both sides blocked or unavailable; rear clear+available
 *   NAV_STOP       – all required readings unavailable or all directions blocked
 *
 * Interaction:
 *   Called by RobotController with distances from UltrasonicSensor and
 *   availability flags from distanceAvailable().
 */

#ifndef NAVIGATION_H
#define NAVIGATION_H

#include "config.h"

enum NavState {
    NAV_IDLE,
    NAV_FORWARD,
    NAV_TURN_LEFT,
    NAV_TURN_RIGHT,
    NAV_BACKWARD,
    NAV_STOP
};

class NavigationManager {
private:
    NavState _state;

    bool isBlocked(bool available, float d) const {
        if (!available) return true;
        return d < OBSTACLE_THRESHOLD_CM;
    }

public:
    NavigationManager() : _state(NAV_IDLE) {}

    /*
     * update() – compute next NavState from sensor readings and availability flags.
     *
     * Parameters (all distances in cm):
     *   frontOk  – true if front sensor has at least one valid reading
     *   front    – front averaged distance
     *   rearOk   – true if rear  sensor has at least one valid reading
     *   rear     – rear  averaged distance
     *   leftOk   – true if left  sensor has at least one valid reading
     *   left     – left  averaged distance
     *   rightOk  – true if right sensor has at least one valid reading
     *   right    – right averaged distance
     *
     * Fail-safe rules:
     *   - Unavailable reading → treat as blocked (never treat as clear).
     *   - Front unavailable   → NAV_STOP (cannot verify forward safety).
     *   - Side unavailable    → treat that side as blocked when choosing a turn.
     *   - Rear unavailable    → NAV_STOP when backward would otherwise be chosen.
     */
    void update(bool frontOk, float front,
                bool rearOk,  float rear,
                bool leftOk,  float left,
                bool rightOk, float right) {

        bool fwd = isBlocked(frontOk, front);
        bool bwd = isBlocked(rearOk,  rear);
        bool lft = isBlocked(leftOk,  left);
        bool rgt = isBlocked(rightOk, right);

        if (!fwd) {
            _state = NAV_FORWARD;
        } else if (!lft && !rgt) {
            _state = NAV_TURN_LEFT;
        } else if (!lft) {
            _state = NAV_TURN_LEFT;
        } else if (!rgt) {
            _state = NAV_TURN_RIGHT;
        } else if (!bwd) {
            _state = NAV_BACKWARD;
        } else {
            _state = NAV_STOP;
        }
    }

    /*
     * reset() – return to NAV_IDLE.
     * Call when leaving AUTO mode so stale state is not acted upon.
     */
    void reset() {
        _state = NAV_IDLE;
    }

    NavState getState() const { return _state; }

    String stateToString() const {
        switch (_state) {
            case NAV_IDLE:        return "idle";
            case NAV_FORWARD:     return "forward";
            case NAV_TURN_LEFT:   return "turn_left";
            case NAV_TURN_RIGHT:  return "turn_right";
            case NAV_BACKWARD:    return "backward";
            case NAV_STOP:        return "stop";
            default:              return "unknown";
        }
    }
};

#endif
