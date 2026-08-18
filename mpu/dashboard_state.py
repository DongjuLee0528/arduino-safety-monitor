"""
Dashboard state layer for the helmet-inspection robot.

Provides a single thread-safe state repository (DashboardState) that holds
all data needed by the App Lab dashboard:
  - Robot connection status
  - Operating mode (AUTO / MANUAL / UNKNOWN)
  - Current movement state
  - Four ultrasonic distances
  - Helmet-detection result
  - Daily statistics (inspected, helmet, no-helmet, warnings; auto-reset at Korea local date rollover)
  - Bounded event log placeholder (structure + storage; no wiring)

Usage:
    state = DashboardState()
    state.update_connection(online=True)
    state.update_detection(worker_present=True, helmet_result=HelmetResult.NO_HELMET)
    state.update_statistics(inspected_delta=1, helmet_delta=0, no_helmet_delta=1)
    snapshot = state.snapshot()   # plain dict, safe to serialize
"""

import math
import copy
import threading
from collections import deque
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RobotConnection(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class RobotMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class MovementState(str, Enum):
    STOPPED = "stopped"
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    NAV_TURNING_LEFT = "nav_turning_left"
    NAV_TURNING_RIGHT = "nav_turning_right"
    NAV_BACKWARD = "nav_backward"
    UNKNOWN = "unknown"


class HelmetResult(str, Enum):
    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    DETECTION = "detection"
    ALERT = "alert"
    CONNECTION = "connection"
    MODE_CHANGE = "mode_change"
    SYSTEM = "system"


class SafetyState(str, Enum):
    SAFE = "safe"
    MOTION_AUTHORIZED = "motion_authorized"
    WARNING = "warning"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_EVENTS = 100
STATISTICS_TIMEZONE = ZoneInfo("Asia/Seoul")

# Sentinel object used by update_distances() to distinguish a deliberately
# omitted argument from an explicitly supplied None (which clears the field).
_UNSET = object()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _statistics_today() -> date:
    """Return the Korea-local calendar date for the current statistics window."""
    return datetime.now(STATISTICS_TIMEZONE).date()


def _utc_isoformat(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to an ISO 8601 string, or return None if dt is None."""
    return dt.isoformat() if dt is not None else None


def _resolve_timestamp(timestamp: Optional[datetime]) -> datetime:
    """
    Return a UTC-normalised datetime.

    If timestamp is None, return the current UTC time.
    If timestamp is supplied, validate and normalise it to UTC.

    Raises:
        TypeError:  If timestamp is not a datetime instance.
        ValueError: If timestamp is naive (no tzinfo or unusable utcoffset).
    """
    if timestamp is None:
        return _now_utc()
    return _validated_and_normalized_ts(timestamp)


def _validated_and_normalized_ts(value: object) -> datetime:
    """
    Validate that value is a timezone-aware datetime and normalise to UTC.

    Raises:
        TypeError:  If value is not a datetime instance.
        ValueError: If value is naive (tzinfo is None or utcoffset() is None).
    """
    if not isinstance(value, datetime):
        raise TypeError(
            f"timestamp must be a datetime instance, got {type(value).__name__}"
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "timestamp must be timezone-aware; naive datetime is not accepted"
        )
    return value.astimezone(timezone.utc)


def _validated_distance(value: object) -> Optional[float]:
    """
    Validate and return a finite numeric distance value as float.

    Accepts int or float (excluding bool).
    Rejects None by design — callers that want to clear use None explicitly
    and bypass this function; callers that supply a value call this function.

    Raises:
        TypeError:  If value is bool, or not int/float.
        ValueError: If value is NaN or infinite.
    """
    if isinstance(value, bool):
        raise TypeError("distance must be int or float, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"distance must be int or float, got {type(value).__name__}"
        )
    f = float(value)
    if math.isnan(f):
        raise ValueError("distance must be a finite number, got NaN")
    if math.isinf(f):
        raise ValueError("distance must be a finite number, got infinity")
    return f


def _validated_bbox(value: object) -> Optional[tuple]:
    """
    Validate and normalise a bounding box value.

    Accepted input: None, or a tuple of exactly four elements where each
    element is a finite int or float (bool excluded).

    Returns None for None input, or a tuple of four floats.

    Raises:
        TypeError:  If value is not None or tuple, or any element is bool or
                    not int/float.
        ValueError: If tuple length != 4, or any element is NaN or infinite.
    """
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise TypeError(
            f"bbox must be a tuple of four numeric values or None, "
            f"got {type(value).__name__}"
        )
    if len(value) != 4:
        raise ValueError(
            f"bbox must have exactly 4 elements, got {len(value)}"
        )
    result = []
    for i, elem in enumerate(value):
        if isinstance(elem, bool):
            raise TypeError(
                f"bbox element {i} must be int or float, not bool"
            )
        if not isinstance(elem, (int, float)):
            raise TypeError(
                f"bbox element {i} must be int or float, "
                f"got {type(elem).__name__}"
            )
        f = float(elem)
        if math.isnan(f):
            raise ValueError(f"bbox element {i} must be finite, got NaN")
        if math.isinf(f):
            raise ValueError(f"bbox element {i} must be finite, got infinity")
        result.append(f)
    return tuple(result)


def _validated_confidence(value: object, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be int, float, or None")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{field_name} must be finite")
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return f


def _validated_detection_entry(value: object) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"detection entry must be dict, got {type(value).__name__}")

    bbox = value.get("person_bbox")
    if isinstance(bbox, list):
        bbox = tuple(bbox)
    validated_bbox = _validated_bbox(bbox)
    if validated_bbox is None:
        raise ValueError("person_bbox is required")

    helmet_result = value.get("helmet_result", HelmetResult.UNKNOWN.value)
    if isinstance(helmet_result, HelmetResult):
        helmet_result = helmet_result.value
    if helmet_result not in {item.value for item in HelmetResult}:
        raise ValueError(f"invalid helmet_result: {helmet_result!r}")

    return {
        "person_bbox": list(validated_bbox),
        "person_confidence": _validated_confidence(
            value.get("person_confidence"), "person_confidence"
        ),
        "helmet_result": helmet_result,
        "helmet_confidence": _validated_confidence(
            value.get("helmet_confidence"), "helmet_confidence"
        ),
    }


def _validated_detections(value: object) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"detections must be a list, got {type(value).__name__}")
    return [_validated_detection_entry(item) for item in value]


# ---------------------------------------------------------------------------
# DashboardState
# ---------------------------------------------------------------------------

class DashboardState:
    """
    Thread-safe owner of all dashboard state.

    All update_*() methods acquire the internal lock, mutate only the
    relevant section, and release.  snapshot() returns a deep-copied
    plain dict that is safe to serialize to JSON and cannot mutate
    internal state.

    Args:
        max_events: Maximum number of events retained in the bounded log.
    """

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        if max_events < 1:
            raise ValueError(f"max_events must be >= 1, got {max_events}")
        self._lock = threading.Lock()  # Protects all mutable state below from concurrent access
        self._max_events = max_events

        # --- connection: tracks whether the robot MCU is reachable ---
        self._connection: RobotConnection = RobotConnection.OFFLINE
        self._connection_updated_at: Optional[datetime] = None  # UTC timestamp of last status change

        # --- mode: AUTO / MANUAL / UNKNOWN operating mode ---
        self._mode: RobotMode = RobotMode.UNKNOWN

        # --- movement: most recent locomotion command sent to the MCU ---
        self._movement: MovementState = MovementState.STOPPED

        # --- distances in cm from four ultrasonic sensors; None = reading unavailable ---
        self._dist_front: Optional[float] = None
        self._dist_rear: Optional[float] = None
        self._dist_left: Optional[float] = None
        self._dist_right: Optional[float] = None

        self._motor_speed: Optional[int] = None
        self._motion_lease_active: Optional[bool] = None
        self._control_tick_fresh: Optional[bool] = None
        self._command_fresh: Optional[bool] = None
        self._warning_active: Optional[bool] = None
        self._safety_state: SafetyState = SafetyState.UNKNOWN
        self._control_updated_at: Optional[datetime] = None

        # --- detection: result of the latest helmet-classification inference ---
        self._worker_present: bool = False
        self._helmet_result: HelmetResult = HelmetResult.UNKNOWN
        self._detection_updated_at: Optional[datetime] = None  # UTC timestamp of last detection
        # Bounding box stored as tuple(x, y, w, h) of floats, or None when absent
        self._detection_bbox: Optional[tuple] = None
        self._detection_confidence: Optional[float] = None
        self._detections: list = []

        # --- daily statistics: reset automatically at Korea local midnight ---
        self._stats_date: date = _statistics_today()  # KST date of current stats window
        self._stats_inspected: int = 0   # Total workers inspected today
        self._stats_helmet: int = 0      # Workers with helmet today
        self._stats_no_helmet: int = 0   # Workers without helmet today
        self._stats_warnings: int = 0    # AlertManager final no-helmet alerts today

        # --- bounded event log: oldest entry dropped when maxlen is reached ---
        # _events_date tracks the KST calendar day for which these events are valid.
        # Events are cleared when the KST date advances (same boundary as statistics).
        self._events_date: date = _statistics_today()
        self._events: deque = deque(maxlen=max_events)

    # -----------------------------------------------------------------------
    # Update methods
    # -----------------------------------------------------------------------

    def update_connection(
        self,
        online: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update robot connection status.

        Args:
            online:    True if the robot MCU is reachable.
            timestamp: Optional explicit timestamp; must be timezone-aware.
                       Normalised to UTC before storage.  Defaults to now.

        Raises:
            TypeError:  If timestamp is not a datetime instance.
            ValueError: If timestamp is naive.
        """
        ts = _resolve_timestamp(timestamp)   # Resolve or default to current UTC time
        with self._lock:
            self._connection = RobotConnection.ONLINE if online else RobotConnection.OFFLINE
            self._connection_updated_at = ts   # Record when the connection status last changed

    def update_mode(self, mode: RobotMode) -> None:
        """
        Update operating mode.

        Args:
            mode: RobotMode enum value.

        Raises:
            TypeError: If mode is not a RobotMode instance.
        """
        if not isinstance(mode, RobotMode):
            raise TypeError(f"mode must be a RobotMode, got {type(mode).__name__}")
        with self._lock:
            self._mode = mode

    def update_movement(self, movement: MovementState) -> None:
        """
        Update current movement state.

        Args:
            movement: MovementState enum value.

        Raises:
            TypeError: If movement is not a MovementState instance.
        """
        if not isinstance(movement, MovementState):
            raise TypeError(
                f"movement must be a MovementState, got {type(movement).__name__}"
            )
        with self._lock:
            self._movement = movement

    def update_distances(
        self,
        front=_UNSET,
        rear=_UNSET,
        left=_UNSET,
        right=_UNSET,
    ) -> None:
        """
        Update ultrasonic sensor distances (true partial update).

        Omitted arguments preserve their current stored value.
        Passing None explicitly clears that specific distance (marks it
        unavailable).  Passing a finite int or float stores it as float.
        bool values are rejected even though bool is a subclass of int.

        Args:
            front: Front sensor distance in cm, None to clear, or omit to keep.
            rear:  Rear sensor distance in cm, None to clear, or omit to keep.
            left:  Left sensor distance in cm, None to clear, or omit to keep.
            right: Right sensor distance in cm, None to clear, or omit to keep.

        Raises:
            TypeError:  If a supplied value is bool or not int/float/None.
            ValueError: If a supplied numeric value is NaN or infinite.
        """
        new_front = None if front is None else (
            _UNSET if front is _UNSET else _validated_distance(front)
        )
        new_rear = None if rear is None else (
            _UNSET if rear is _UNSET else _validated_distance(rear)
        )
        new_left = None if left is None else (
            _UNSET if left is _UNSET else _validated_distance(left)
        )
        new_right = None if right is None else (
            _UNSET if right is _UNSET else _validated_distance(right)
        )

        with self._lock:
            if new_front is not _UNSET:
                self._dist_front = new_front
            if new_rear is not _UNSET:
                self._dist_rear = new_rear
            if new_left is not _UNSET:
                self._dist_left = new_left
            if new_right is not _UNSET:
                self._dist_right = new_right

    def update_control(
        self,
        *,
        motion_lease_active: Optional[bool] = None,
        control_tick_fresh: Optional[bool] = None,
        command_fresh: Optional[bool] = None,
        warning_active: Optional[bool] = None,
        safety_state: SafetyState = SafetyState.UNKNOWN,
        motor_speed: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        if not isinstance(safety_state, SafetyState):
            raise TypeError(
                f"safety_state must be a SafetyState, got {type(safety_state).__name__}"
            )
        if motor_speed is not None:
            if isinstance(motor_speed, bool) or not isinstance(motor_speed, int):
                raise TypeError("motor_speed must be int or None")
            if motor_speed < 0:
                raise ValueError("motor_speed must be non-negative")
        ts = _resolve_timestamp(timestamp)
        with self._lock:
            self._motion_lease_active = None if motion_lease_active is None else bool(motion_lease_active)
            self._control_tick_fresh = None if control_tick_fresh is None else bool(control_tick_fresh)
            self._command_fresh = None if command_fresh is None else bool(command_fresh)
            self._warning_active = None if warning_active is None else bool(warning_active)
            self._safety_state = safety_state
            self._motor_speed = motor_speed
            self._control_updated_at = ts

    def update_detection(
        self,
        worker_present: bool,
        helmet_result: HelmetResult = HelmetResult.UNKNOWN,
        timestamp: Optional[datetime] = None,
        bbox: Optional[tuple] = None,
        confidence: Optional[float] = None,
        detections: Optional[list] = None,
    ) -> None:
        """
        Update the current detection result.

        Args:
            worker_present: True if at least one worker is detected in frame.
            helmet_result:  HelmetResult enum value.
            timestamp:      Optional explicit timestamp; must be timezone-aware.
                            Normalised to UTC before storage.  Defaults to now.
            bbox:           Optional (x, y, w, h) bounding box as a tuple of
                            exactly four finite int or float values (bool
                            excluded).  No OpenCV objects are accepted.
            confidence:     Optional helmet-classifier confidence for the
                            legacy summary result, in [0.0, 1.0].
            detections:     Optional list of per-person detection dicts.
                            Each entry uses bbox convention [x, y, w, h].

        Raises:
            TypeError:  If helmet_result is not a HelmetResult instance, or
                        bbox is not a tuple/None, or any bbox element is bool
                        or not int/float, or timestamp is not datetime.
            ValueError: If bbox has wrong length or any element is NaN/infinite,
                        or timestamp is naive.
        """
        if not isinstance(helmet_result, HelmetResult):
            raise TypeError(
                f"helmet_result must be a HelmetResult, got {type(helmet_result).__name__}"
            )
        validated_bbox = _validated_bbox(bbox)
        validated_confidence = _validated_confidence(confidence, "confidence")
        validated_detections = _validated_detections(detections)
        ts = _resolve_timestamp(timestamp)
        with self._lock:
            self._worker_present = bool(worker_present)
            self._helmet_result = helmet_result
            self._detection_updated_at = ts
            self._detection_bbox = validated_bbox
            self._detection_confidence = validated_confidence
            self._detections = validated_detections

    def update_statistics(
        self,
        inspected_delta: int = 0,
        helmet_delta: int = 0,
        no_helmet_delta: int = 0,
        warnings_delta: int = 0,
    ) -> None:
        """
        Increment daily statistics counters, resetting them first if the Korea
        local calendar date has advanced since the last reset.

        All delta values must be non-negative integers.

        Args:
            inspected_delta:  Number of newly counted workers to add.
            helmet_delta:     Number of new helmet-compliant workers to add.
            no_helmet_delta:  Number of new non-compliant workers to add.
            warnings_delta:    Number of final AlertManager alert triggers to add.

        Raises:
            TypeError:  If any delta is not an int (bool excluded).
            ValueError: If any delta is negative.
        """
        for name, val in (
            ("inspected_delta", inspected_delta),
            ("helmet_delta", helmet_delta),
            ("no_helmet_delta", no_helmet_delta),
            ("warnings_delta", warnings_delta),
        ):
            if isinstance(val, bool):
                raise TypeError(f"{name} must be int, not bool")
            if not isinstance(val, int):
                raise TypeError(f"{name} must be int, got {type(val).__name__}")
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        with self._lock:
            self._rollover_statistics_if_needed()
            self._stats_inspected += inspected_delta
            self._stats_helmet += helmet_delta
            self._stats_no_helmet += no_helmet_delta
            self._stats_warnings += warnings_delta

    def _rollover_statistics_if_needed(self) -> None:
        """
        Reset daily statistics if the Korea-local date has changed.

        Caller must hold self._lock.
        """
        today = _statistics_today()
        if today != self._stats_date:
            self._stats_date = today
            self._stats_inspected = 0
            self._stats_helmet = 0
            self._stats_no_helmet = 0
            self._stats_warnings = 0

    def _rollover_events_if_needed(self) -> None:
        """
        Clear the event log if the Korea-local date has advanced since the
        last event was recorded.

        Uses the same Asia/Seoul date boundary as statistics rollover so that
        Recent Events and Today Summary always reflect the same calendar day.

        Caller must hold self._lock.
        """
        today = _statistics_today()
        if today != self._events_date:
            self._events_date = today
            self._events.clear()

    def append_event(
        self,
        event_type: EventType,
        message: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Append a structured event to the bounded event log.

        When the log is full, the oldest event is discarded automatically
        (deque maxlen behaviour).

        Args:
            event_type: EventType enum value.
            message:    Human-readable description of the event.
            timestamp:  Optional explicit timestamp; must be timezone-aware.
                        Normalised to UTC before storage.  Defaults to now.

        Raises:
            TypeError:  If event_type is not an EventType instance, or
                        timestamp is not a datetime instance.
            ValueError: If timestamp is naive.
        """
        if not isinstance(event_type, EventType):
            raise TypeError(
                f"event_type must be an EventType, got {type(event_type).__name__}"
            )
        ts = _resolve_timestamp(timestamp)
        event = {
            "timestamp": ts.isoformat(),
            "event_type": event_type.value,
            "message": str(message),
        }
        with self._lock:
            self._rollover_events_if_needed()
            self._events.append(event)

    # -----------------------------------------------------------------------
    # Snapshot
    # -----------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a deep-copied plain dict of the entire current state.

        The returned dict is independent of internal state: mutating it
        has no effect on this DashboardState instance.  All values are
        JSON-compatible types (str, float, int, bool, list, dict, None).

        Returns:
            dict with keys: connection, mode, movement, distances,
            detection, statistics, events.
        """
        with self._lock:   # Hold the lock only while reading; deepcopy runs after release
            self._rollover_statistics_if_needed()
            self._rollover_events_if_needed()
            raw = {
                "connection": {
                    "status": self._connection.value,
                    "updated_at": _utc_isoformat(self._connection_updated_at),
                },
                "mode": self._mode.value,
                "movement": self._movement.value,
                "control": {
                    "motor_speed": self._motor_speed,
                    "motion_lease_active": self._motion_lease_active,
                    "control_tick_fresh": self._control_tick_fresh,
                    "command_fresh": self._command_fresh,
                    "warning_active": self._warning_active,
                    "safety_state": self._safety_state.value,
                    "updated_at": _utc_isoformat(self._control_updated_at),
                },
                "distances": {
                    "front": self._dist_front,
                    "rear": self._dist_rear,
                    "left": self._dist_left,
                    "right": self._dist_right,
                },
                "detection": {
                    "worker_present": self._worker_present,
                    "helmet_result": self._helmet_result.value,
                    "updated_at": _utc_isoformat(self._detection_updated_at),
                    # Convert tuple to list for JSON serialisation compatibility
                    "bbox": list(self._detection_bbox) if self._detection_bbox is not None else None,
                    "confidence": self._detection_confidence,
                    "bbox_format": "xywh",
                    "detections": copy.deepcopy(self._detections),
                },
                "statistics": {
                    "date": self._stats_date.isoformat(),
                    "inspected": self._stats_inspected,
                    "helmet": self._stats_helmet,
                    "no_helmet": self._stats_no_helmet,
                    "warnings": self._stats_warnings,
                },
                "events": list(self._events),  # Snapshot the deque as a plain list
            }
        return copy.deepcopy(raw)  # Deep copy ensures callers cannot mutate internal state
