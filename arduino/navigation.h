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
    NavState _state;  // Current navigation decision; read by RobotController to drive the motors

    /*
     * isBlocked() – returns true if the direction should be treated as impassable.
     * An unavailable reading is always considered blocked (fail-safe).
     * An available reading is blocked when the distance is below the obstacle threshold.
     */
    bool isBlocked(bool available, float d) const {
        if (!available) return true;               // Missing data -> assume obstacle present
        return d < OBSTACLE_THRESHOLD_CM;          // Too close -> treat as blocked
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

        // Evaluate each direction with unknown readings folded into "blocked".
        bool fwd = isBlocked(frontOk, front);   // true = front path is obstructed or unknown
        bool bwd = isBlocked(rearOk,  rear);    // true = rear  path is obstructed or unknown
        bool lft = isBlocked(leftOk,  left);    // true = left  path is obstructed or unknown
        bool rgt = isBlocked(rightOk, right);   // true = right path is obstructed or unknown

        if (!fwd) {
            _state = NAV_FORWARD;       // Front clear -> move forward (preferred direction)
        } else if (!lft && !rgt) {
            _state = NAV_TURN_LEFT;     // Both side paths are clear -> prefer left turn
        } else if (!lft) {
            _state = NAV_TURN_LEFT;     // Only left open -> turn left
        } else if (!rgt) {
            _state = NAV_TURN_RIGHT;    // Only right open -> turn right
        } else if (!bwd) {
            _state = NAV_BACKWARD;      // Front and sides all blocked; rear available -> reverse
        } else {
            _state = NAV_STOP;          // All directions blocked or unavailable -> stop safely
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
