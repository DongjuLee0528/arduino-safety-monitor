"""
Tests for mpu.dashboard_state.

Coverage:
  1. Default state values
  2. Updating connection status
  3. Updating mode and movement state
  4. Updating all four distances
  5. Updating detection state
  6. Snapshot isolation from internal state
  7. Concurrent / thread-safe access
  8. Event append and bounded event retention
  9. JSON-compatible snapshot output
 10. Invalid enum / value handling
 11. bbox validation (blocking issue 1)
 12. Timestamp validation and UTC normalisation (blocking issue 2)
 13. Partial distance update contract (blocking issue 3)
"""

import json
import threading
import unittest
from datetime import datetime, timezone, timedelta

from mpu.dashboard_state import (
    DEFAULT_MAX_EVENTS,
    DashboardState,
    EventType,
    HelmetResult,
    MovementState,
    RobotConnection,
    RobotMode,
)

_UTC = timezone.utc


class TestDefaultState(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_connection_defaults_offline(self):
        snap = self.state.snapshot()
        self.assertEqual(snap["connection"]["status"], RobotConnection.OFFLINE.value)

    def test_connection_updated_at_defaults_none(self):
        snap = self.state.snapshot()
        self.assertIsNone(snap["connection"]["updated_at"])

    def test_mode_defaults_unknown(self):
        snap = self.state.snapshot()
        self.assertEqual(snap["mode"], RobotMode.UNKNOWN.value)

    def test_movement_defaults_stopped(self):
        snap = self.state.snapshot()
        self.assertEqual(snap["movement"], MovementState.STOPPED.value)

    def test_distances_default_none(self):
        snap = self.state.snapshot()
        d = snap["distances"]
        self.assertIsNone(d["front"])
        self.assertIsNone(d["rear"])
        self.assertIsNone(d["left"])
        self.assertIsNone(d["right"])

    def test_detection_defaults(self):
        snap = self.state.snapshot()
        det = snap["detection"]
        self.assertFalse(det["worker_present"])
        self.assertEqual(det["helmet_result"], HelmetResult.UNKNOWN.value)
        self.assertIsNone(det["updated_at"])
        self.assertIsNone(det["bbox"])

    def test_statistics_defaults(self):
        snap = self.state.snapshot()
        stats = snap["statistics"]
        self.assertEqual(stats["inspected"], 0)
        self.assertEqual(stats["helmet"], 0)
        self.assertEqual(stats["no_helmet"], 0)
        self.assertIsInstance(stats["date"], str)

    def test_events_defaults_empty(self):
        snap = self.state.snapshot()
        self.assertEqual(snap["events"], [])

    def test_default_max_events_constant_positive(self):
        self.assertGreater(DEFAULT_MAX_EVENTS, 0)


class TestConnectionUpdate(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_set_online(self):
        self.state.update_connection(online=True)
        snap = self.state.snapshot()
        self.assertEqual(snap["connection"]["status"], "online")

    def test_set_offline(self):
        self.state.update_connection(online=True)
        self.state.update_connection(online=False)
        snap = self.state.snapshot()
        self.assertEqual(snap["connection"]["status"], "offline")

    def test_timestamp_recorded(self):
        ts = datetime(2026, 7, 29, 12, 0, 0, tzinfo=_UTC)
        self.state.update_connection(online=True, timestamp=ts)
        snap = self.state.snapshot()
        self.assertIn("2026-07-29", snap["connection"]["updated_at"])

    def test_timestamp_defaults_to_now(self):
        self.state.update_connection(online=True)
        snap = self.state.snapshot()
        self.assertIsNotNone(snap["connection"]["updated_at"])

    def test_toggle_does_not_affect_other_fields(self):
        self.state.update_mode(RobotMode.AUTO)
        self.state.update_connection(online=True)
        snap = self.state.snapshot()
        self.assertEqual(snap["mode"], "auto")


class TestModeAndMovement(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_set_auto_mode(self):
        self.state.update_mode(RobotMode.AUTO)
        self.assertEqual(self.state.snapshot()["mode"], "auto")

    def test_set_manual_mode(self):
        self.state.update_mode(RobotMode.MANUAL)
        self.assertEqual(self.state.snapshot()["mode"], "manual")

    def test_set_unknown_mode(self):
        self.state.update_mode(RobotMode.AUTO)
        self.state.update_mode(RobotMode.UNKNOWN)
        self.assertEqual(self.state.snapshot()["mode"], "unknown")

    def test_set_forward(self):
        self.state.update_movement(MovementState.FORWARD)
        self.assertEqual(self.state.snapshot()["movement"], "forward")

    def test_set_backward(self):
        self.state.update_movement(MovementState.BACKWARD)
        self.assertEqual(self.state.snapshot()["movement"], "backward")

    def test_set_left(self):
        self.state.update_movement(MovementState.LEFT)
        self.assertEqual(self.state.snapshot()["movement"], "left")

    def test_set_right(self):
        self.state.update_movement(MovementState.RIGHT)
        self.assertEqual(self.state.snapshot()["movement"], "right")

    def test_nav_turning_left(self):
        self.state.update_movement(MovementState.NAV_TURNING_LEFT)
        self.assertEqual(self.state.snapshot()["movement"], "nav_turning_left")

    def test_nav_turning_right(self):
        self.state.update_movement(MovementState.NAV_TURNING_RIGHT)
        self.assertEqual(self.state.snapshot()["movement"], "nav_turning_right")

    def test_nav_backward(self):
        self.state.update_movement(MovementState.NAV_BACKWARD)
        self.assertEqual(self.state.snapshot()["movement"], "nav_backward")

    def test_mode_update_does_not_touch_movement(self):
        self.state.update_movement(MovementState.FORWARD)
        self.state.update_mode(RobotMode.MANUAL)
        self.assertEqual(self.state.snapshot()["movement"], "forward")


class TestDistanceUpdate(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_all_four_distances(self):
        self.state.update_distances(front=10.5, rear=20.0, left=15.3, right=8.1)
        d = self.state.snapshot()["distances"]
        self.assertAlmostEqual(d["front"], 10.5)
        self.assertAlmostEqual(d["rear"], 20.0)
        self.assertAlmostEqual(d["left"], 15.3)
        self.assertAlmostEqual(d["right"], 8.1)

    def test_none_distance_stays_none(self):
        self.state.update_distances(front=None, rear=None, left=None, right=None)
        d = self.state.snapshot()["distances"]
        self.assertIsNone(d["front"])
        self.assertIsNone(d["rear"])
        self.assertIsNone(d["left"])
        self.assertIsNone(d["right"])

    def test_inf_raises(self):
        with self.assertRaises(ValueError):
            self.state.update_distances(front=float("inf"))

    def test_nan_raises(self):
        with self.assertRaises(ValueError):
            self.state.update_distances(right=float("nan"))

    def test_zero_distance_valid(self):
        self.state.update_distances(front=0.0)
        self.assertAlmostEqual(self.state.snapshot()["distances"]["front"], 0.0)

    def test_large_distance_valid(self):
        self.state.update_distances(rear=300.0)
        self.assertAlmostEqual(self.state.snapshot()["distances"]["rear"], 300.0)

    def test_distance_update_does_not_touch_mode(self):
        self.state.update_mode(RobotMode.AUTO)
        self.state.update_distances(front=50.0)
        self.assertEqual(self.state.snapshot()["mode"], "auto")


class TestDetectionUpdate(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_worker_present_helmet(self):
        self.state.update_detection(worker_present=True, helmet_result=HelmetResult.HELMET)
        det = self.state.snapshot()["detection"]
        self.assertTrue(det["worker_present"])
        self.assertEqual(det["helmet_result"], "helmet")

    def test_worker_present_no_helmet(self):
        self.state.update_detection(worker_present=True, helmet_result=HelmetResult.NO_HELMET)
        self.assertEqual(self.state.snapshot()["detection"]["helmet_result"], "no_helmet")

    def test_no_worker_clears_result(self):
        self.state.update_detection(worker_present=True, helmet_result=HelmetResult.NO_HELMET)
        self.state.update_detection(worker_present=False, helmet_result=HelmetResult.UNKNOWN)
        det = self.state.snapshot()["detection"]
        self.assertFalse(det["worker_present"])
        self.assertEqual(det["helmet_result"], "unknown")

    def test_timestamp_stored(self):
        ts = datetime(2026, 1, 15, 9, 30, 0, tzinfo=_UTC)
        self.state.update_detection(worker_present=True, helmet_result=HelmetResult.HELMET, timestamp=ts)
        snap = self.state.snapshot()["detection"]
        self.assertIn("2026-01-15", snap["updated_at"])

    def test_bbox_stored_as_list(self):
        self.state.update_detection(
            worker_present=True,
            helmet_result=HelmetResult.HELMET,
            bbox=(10, 20, 80, 160),
        )
        det = self.state.snapshot()["detection"]
        self.assertEqual(det["bbox"], [10, 20, 80, 160])

    def test_bbox_none_stored_as_none(self):
        self.state.update_detection(worker_present=False)
        self.assertIsNone(self.state.snapshot()["detection"]["bbox"])

    def test_detection_update_does_not_touch_distances(self):
        self.state.update_distances(front=55.0)
        self.state.update_detection(worker_present=True, helmet_result=HelmetResult.HELMET)
        self.assertAlmostEqual(self.state.snapshot()["distances"]["front"], 55.0)


class TestSnapshotIsolation(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_mutating_snapshot_dict_does_not_affect_state(self):
        snap1 = self.state.snapshot()
        snap1["mode"] = "mutated"
        snap2 = self.state.snapshot()
        self.assertEqual(snap2["mode"], "unknown")

    def test_mutating_snapshot_distances_does_not_affect_state(self):
        self.state.update_distances(front=42.0)
        snap = self.state.snapshot()
        snap["distances"]["front"] = 999.0
        self.assertAlmostEqual(self.state.snapshot()["distances"]["front"], 42.0)

    def test_mutating_snapshot_events_list_does_not_affect_state(self):
        self.state.append_event(EventType.SYSTEM, "boot")
        snap = self.state.snapshot()
        snap["events"].clear()
        self.assertEqual(len(self.state.snapshot()["events"]), 1)

    def test_mutating_snapshot_event_dict_does_not_affect_state(self):
        self.state.append_event(EventType.SYSTEM, "boot")
        snap = self.state.snapshot()
        snap["events"][0]["message"] = "tampered"
        second_snap = self.state.snapshot()
        self.assertEqual(second_snap["events"][0]["message"], "boot")

    def test_two_snapshots_are_independent(self):
        snap_a = self.state.snapshot()
        self.state.update_mode(RobotMode.AUTO)
        snap_b = self.state.snapshot()
        self.assertEqual(snap_a["mode"], "unknown")
        self.assertEqual(snap_b["mode"], "auto")


class TestConcurrentAccess(unittest.TestCase):
    def test_concurrent_connection_updates_no_exception(self):
        state = DashboardState()
        errors = []

        def toggle(n):
            try:
                for i in range(n):
                    state.update_connection(online=(i % 2 == 0))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=toggle, args=(200,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        snap = state.snapshot()
        self.assertIn(snap["connection"]["status"], ("online", "offline"))

    def test_concurrent_distance_and_detection_updates(self):
        state = DashboardState()
        errors = []

        def update_distances():
            try:
                for i in range(100):
                    state.update_distances(front=float(i), rear=float(i))
            except Exception as exc:
                errors.append(exc)

        def update_detection():
            try:
                for i in range(100):
                    state.update_detection(
                        worker_present=True,
                        helmet_result=HelmetResult.HELMET,
                    )
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=update_distances)
        t2 = threading.Thread(target=update_detection)
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertEqual(errors, [])

    def test_concurrent_snapshot_calls_no_exception(self):
        state = DashboardState()
        errors = []

        def read():
            try:
                for _ in range(50):
                    _ = state.snapshot()
            except Exception as exc:
                errors.append(exc)

        def write():
            try:
                for i in range(50):
                    state.update_connection(online=(i % 2 == 0))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=read) for _ in range(4)]
        threads += [threading.Thread(target=write) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])


class TestEventLog(unittest.TestCase):
    def test_append_single_event(self):
        state = DashboardState()
        state.append_event(EventType.SYSTEM, "started")
        events = state.snapshot()["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "system")
        self.assertEqual(events[0]["message"], "started")

    def test_event_has_timestamp(self):
        state = DashboardState()
        state.append_event(EventType.CONNECTION, "connected")
        event = state.snapshot()["events"][0]
        self.assertIn("timestamp", event)
        self.assertIsInstance(event["timestamp"], str)

    def test_explicit_timestamp_in_event(self):
        state = DashboardState()
        ts = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        state.append_event(EventType.ALERT, "alert fired", timestamp=ts)
        event = state.snapshot()["events"][0]
        self.assertIn("2026-06-01", event["timestamp"])

    def test_event_ordering_fifo(self):
        state = DashboardState()
        for i in range(3):
            state.append_event(EventType.SYSTEM, f"msg{i}")
        events = state.snapshot()["events"]
        self.assertEqual([e["message"] for e in events], ["msg0", "msg1", "msg2"])

    def test_bounded_retention_drops_oldest(self):
        state = DashboardState(max_events=3)
        for i in range(5):
            state.append_event(EventType.SYSTEM, f"event{i}")
        events = state.snapshot()["events"]
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["message"], "event2")
        self.assertEqual(events[-1]["message"], "event4")

    def test_default_max_events_respected(self):
        state = DashboardState()
        for i in range(DEFAULT_MAX_EVENTS + 10):
            state.append_event(EventType.SYSTEM, str(i))
        self.assertEqual(len(state.snapshot()["events"]), DEFAULT_MAX_EVENTS)

    def test_all_event_types(self):
        state = DashboardState()
        for et in EventType:
            state.append_event(et, "test")
        events = state.snapshot()["events"]
        recorded_types = {e["event_type"] for e in events}
        self.assertEqual(recorded_types, {et.value for et in EventType})

    def test_max_events_one(self):
        state = DashboardState(max_events=1)
        state.append_event(EventType.SYSTEM, "first")
        state.append_event(EventType.SYSTEM, "second")
        events = state.snapshot()["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["message"], "second")

    def test_max_events_zero_raises(self):
        with self.assertRaises(ValueError):
            DashboardState(max_events=0)

    def test_max_events_negative_raises(self):
        with self.assertRaises(ValueError):
            DashboardState(max_events=-5)


class TestJsonCompatibleSnapshot(unittest.TestCase):
    def test_default_snapshot_json_serializable(self):
        state = DashboardState()
        snap = state.snapshot()
        serialized = json.dumps(snap)
        self.assertIsInstance(serialized, str)

    def test_full_snapshot_json_serializable(self):
        state = DashboardState()
        state.update_connection(online=True)
        state.update_mode(RobotMode.AUTO)
        state.update_movement(MovementState.FORWARD)
        state.update_distances(front=30.0, rear=45.5, left=20.1, right=12.7)
        state.update_detection(
            worker_present=True,
            helmet_result=HelmetResult.NO_HELMET,
            bbox=(5, 10, 100, 200),
        )
        state.append_event(EventType.ALERT, "no helmet detected")
        snap = state.snapshot()
        serialized = json.dumps(snap)
        restored = json.loads(serialized)
        self.assertEqual(restored["connection"]["status"], "online")
        self.assertEqual(restored["mode"], "auto")
        self.assertEqual(restored["movement"], "forward")
        self.assertAlmostEqual(restored["distances"]["front"], 30.0)
        self.assertTrue(restored["detection"]["worker_present"])
        self.assertEqual(restored["detection"]["helmet_result"], "no_helmet")
        self.assertEqual(restored["detection"]["bbox"], [5, 10, 100, 200])
        self.assertEqual(len(restored["events"]), 1)

    def test_snapshot_with_none_distances_json_serializable(self):
        state = DashboardState()
        state.update_distances(front=None, rear=None, left=None, right=None)
        snap = state.snapshot()
        json.dumps(snap)

    def test_snapshot_values_are_primitive_types(self):
        state = DashboardState()
        state.update_distances(front=10.0)
        state.append_event(EventType.CONNECTION, "test")
        snap = state.snapshot()
        self.assertIsInstance(snap["mode"], str)
        self.assertIsInstance(snap["movement"], str)
        self.assertIsInstance(snap["distances"]["front"], float)
        self.assertIsInstance(snap["detection"]["worker_present"], bool)
        self.assertIsInstance(snap["events"], list)


class TestInvalidEnumHandling(unittest.TestCase):
    def test_invalid_mode_string_raises(self):
        state = DashboardState()
        with self.assertRaises(TypeError):
            state.update_mode("auto")

    def test_invalid_mode_none_raises(self):
        state = DashboardState()
        with self.assertRaises(TypeError):
            state.update_mode(None)

    def test_invalid_movement_string_raises(self):
        state = DashboardState()
        with self.assertRaises(TypeError):
            state.update_movement("forward")

    def test_invalid_helmet_result_string_raises(self):
        state = DashboardState()
        with self.assertRaises(TypeError):
            state.update_detection(worker_present=True, helmet_result="helmet")

    def test_invalid_helmet_result_none_raises(self):
        state = DashboardState()
        with self.assertRaises(TypeError):
            state.update_detection(worker_present=True, helmet_result=None)

    def test_invalid_event_type_raises(self):
        state = DashboardState()
        with self.assertRaises(TypeError):
            state.append_event("system", "msg")

    def test_non_numeric_distance_raises(self):
        state = DashboardState()
        with self.assertRaises(TypeError):
            state.update_distances(front="not_a_number")

    def test_invalid_bbox_type_raises(self):
        state = DashboardState()
        with self.assertRaises(TypeError):
            state.update_detection(worker_present=True, bbox=[1, 2, 3, 4])

    def test_state_unchanged_after_failed_update(self):
        state = DashboardState()
        state.update_mode(RobotMode.AUTO)
        try:
            state.update_mode("bad")
        except TypeError:
            pass
        self.assertEqual(state.snapshot()["mode"], "auto")


# ---------------------------------------------------------------------------
# Blocking issue 1 — bbox validation
# ---------------------------------------------------------------------------

class TestBboxValidation(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def _set_bbox(self, bbox):
        self.state.update_detection(worker_present=True, helmet_result=HelmetResult.HELMET, bbox=bbox)

    def test_valid_integer_bbox(self):
        self._set_bbox((10, 20, 80, 160))
        self.assertEqual(self.state.snapshot()["detection"]["bbox"], [10.0, 20.0, 80.0, 160.0])

    def test_valid_float_bbox(self):
        self._set_bbox((1.5, 2.5, 80.0, 160.0))
        snap = self.state.snapshot()["detection"]["bbox"]
        self.assertAlmostEqual(snap[0], 1.5)
        self.assertAlmostEqual(snap[1], 2.5)

    def test_none_bbox(self):
        self._set_bbox(None)
        self.assertIsNone(self.state.snapshot()["detection"]["bbox"])

    def test_wrong_length_three_raises(self):
        with self.assertRaises(ValueError):
            self._set_bbox((1, 2, 3))

    def test_wrong_length_five_raises(self):
        with self.assertRaises(ValueError):
            self._set_bbox((1, 2, 3, 4, 5))

    def test_wrong_length_empty_raises(self):
        with self.assertRaises(ValueError):
            self._set_bbox(())

    def test_string_element_raises(self):
        with self.assertRaises(TypeError):
            self._set_bbox(("a", 2, 3, 4))

    def test_boolean_element_raises(self):
        with self.assertRaises(TypeError):
            self._set_bbox((True, 2, 3, 4))

    def test_nested_list_element_raises(self):
        with self.assertRaises(TypeError):
            self._set_bbox(([1, 2], 3, 4, 5))

    def test_nan_element_raises(self):
        with self.assertRaises(ValueError):
            self._set_bbox((float("nan"), 2, 3, 4))

    def test_positive_infinity_raises(self):
        with self.assertRaises(ValueError):
            self._set_bbox((float("inf"), 2, 3, 4))

    def test_negative_infinity_raises(self):
        with self.assertRaises(ValueError):
            self._set_bbox((float("-inf"), 2, 3, 4))

    def test_json_dumps_succeeds_after_integer_bbox(self):
        self._set_bbox((0, 0, 640, 480))
        json.dumps(self.state.snapshot())

    def test_json_dumps_succeeds_after_float_bbox(self):
        self._set_bbox((0.5, 1.5, 640.0, 480.0))
        json.dumps(self.state.snapshot())

    def test_json_dumps_succeeds_after_none_bbox(self):
        self._set_bbox(None)
        json.dumps(self.state.snapshot())

    def test_bbox_list_type_raises(self):
        with self.assertRaises(TypeError):
            self._set_bbox([1, 2, 3, 4])

    def test_state_unchanged_after_invalid_bbox(self):
        self._set_bbox((10, 20, 30, 40))
        try:
            self._set_bbox((float("nan"), 2, 3, 4))
        except ValueError:
            pass
        self.assertEqual(self.state.snapshot()["detection"]["bbox"], [10.0, 20.0, 30.0, 40.0])


# ---------------------------------------------------------------------------
# Blocking issue 2 — timestamp validation and UTC normalisation
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_TZ_PLUS5 = timezone(timedelta(hours=5))


class TestTimestampValidation(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_auto_generated_connection_ts_is_utc_aware(self):
        self.state.update_connection(online=True)
        ts_str = self.state.snapshot()["connection"]["updated_at"]
        self.assertIsNotNone(ts_str)
        self.assertIn("+00:00", ts_str)

    def test_auto_generated_detection_ts_is_utc_aware(self):
        self.state.update_detection(worker_present=True)
        ts_str = self.state.snapshot()["detection"]["updated_at"]
        self.assertIsNotNone(ts_str)
        self.assertIn("+00:00", ts_str)

    def test_auto_generated_event_ts_is_utc_aware(self):
        self.state.append_event(EventType.SYSTEM, "boot")
        ts_str = self.state.snapshot()["events"][0]["timestamp"]
        self.assertIn("+00:00", ts_str)

    def test_aware_non_utc_normalised_to_utc_connection(self):
        ts = datetime(2026, 7, 29, 17, 0, 0, tzinfo=_TZ_PLUS5)
        self.state.update_connection(online=True, timestamp=ts)
        ts_str = self.state.snapshot()["connection"]["updated_at"]
        self.assertIn("+00:00", ts_str)
        self.assertIn("2026-07-29T12:00:00", ts_str)

    def test_aware_non_utc_normalised_to_utc_detection(self):
        ts = datetime(2026, 7, 29, 17, 0, 0, tzinfo=_TZ_PLUS5)
        self.state.update_detection(worker_present=True, timestamp=ts)
        ts_str = self.state.snapshot()["detection"]["updated_at"]
        self.assertIn("+00:00", ts_str)
        self.assertIn("2026-07-29T12:00:00", ts_str)

    def test_aware_non_utc_normalised_to_utc_event(self):
        ts = datetime(2026, 7, 29, 17, 0, 0, tzinfo=_TZ_PLUS5)
        self.state.append_event(EventType.SYSTEM, "msg", timestamp=ts)
        ts_str = self.state.snapshot()["events"][0]["timestamp"]
        self.assertIn("+00:00", ts_str)
        self.assertIn("2026-07-29T12:00:00", ts_str)

    def test_naive_datetime_rejected_connection(self):
        naive = datetime(2026, 7, 29, 12, 0, 0)
        with self.assertRaises(ValueError):
            self.state.update_connection(online=True, timestamp=naive)

    def test_naive_datetime_rejected_detection(self):
        naive = datetime(2026, 7, 29, 12, 0, 0)
        with self.assertRaises(ValueError):
            self.state.update_detection(worker_present=True, timestamp=naive)

    def test_naive_datetime_rejected_event(self):
        naive = datetime(2026, 7, 29, 12, 0, 0)
        with self.assertRaises(ValueError):
            self.state.append_event(EventType.SYSTEM, "msg", timestamp=naive)

    def test_non_datetime_rejected_connection(self):
        with self.assertRaises(TypeError):
            self.state.update_connection(online=True, timestamp="2026-07-29T12:00:00Z")

    def test_non_datetime_rejected_detection(self):
        with self.assertRaises(TypeError):
            self.state.update_detection(worker_present=True, timestamp=1234567890)

    def test_non_datetime_rejected_event(self):
        with self.assertRaises(TypeError):
            self.state.append_event(EventType.SYSTEM, "msg", timestamp=0.0)

    def test_snapshot_connection_ts_has_explicit_utc_offset(self):
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=_UTC)
        self.state.update_connection(online=True, timestamp=ts)
        ts_str = self.state.snapshot()["connection"]["updated_at"]
        self.assertTrue(ts_str.endswith("+00:00"))

    def test_snapshot_detection_ts_has_explicit_utc_offset(self):
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=_UTC)
        self.state.update_detection(worker_present=True, timestamp=ts)
        ts_str = self.state.snapshot()["detection"]["updated_at"]
        self.assertTrue(ts_str.endswith("+00:00"))

    def test_snapshot_event_ts_has_explicit_utc_offset(self):
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=_UTC)
        self.state.append_event(EventType.SYSTEM, "msg", timestamp=ts)
        ts_str = self.state.snapshot()["events"][0]["timestamp"]
        self.assertTrue(ts_str.endswith("+00:00"))


# ---------------------------------------------------------------------------
# Blocking issue 3 — partial distance update contract
# ---------------------------------------------------------------------------

class TestPartialDistanceUpdate(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()
        self.state.update_distances(front=10.0, rear=20.0, left=30.0, right=40.0)

    def _distances(self):
        return self.state.snapshot()["distances"]

    def test_update_only_front_preserves_others(self):
        self.state.update_distances(front=99.0)
        d = self._distances()
        self.assertAlmostEqual(d["front"], 99.0)
        self.assertAlmostEqual(d["rear"], 20.0)
        self.assertAlmostEqual(d["left"], 30.0)
        self.assertAlmostEqual(d["right"], 40.0)

    def test_update_only_left_preserves_others(self):
        self.state.update_distances(left=55.0)
        d = self._distances()
        self.assertAlmostEqual(d["front"], 10.0)
        self.assertAlmostEqual(d["rear"], 20.0)
        self.assertAlmostEqual(d["left"], 55.0)
        self.assertAlmostEqual(d["right"], 40.0)

    def test_explicit_none_clears_only_that_field(self):
        self.state.update_distances(rear=None)
        d = self._distances()
        self.assertAlmostEqual(d["front"], 10.0)
        self.assertIsNone(d["rear"])
        self.assertAlmostEqual(d["left"], 30.0)
        self.assertAlmostEqual(d["right"], 40.0)

    def test_multiple_supplied_fields_update_together(self):
        self.state.update_distances(front=1.0, right=2.0)
        d = self._distances()
        self.assertAlmostEqual(d["front"], 1.0)
        self.assertAlmostEqual(d["rear"], 20.0)
        self.assertAlmostEqual(d["left"], 30.0)
        self.assertAlmostEqual(d["right"], 2.0)

    def test_no_args_changes_nothing(self):
        self.state.update_distances()
        d = self._distances()
        self.assertAlmostEqual(d["front"], 10.0)
        self.assertAlmostEqual(d["rear"], 20.0)
        self.assertAlmostEqual(d["left"], 30.0)
        self.assertAlmostEqual(d["right"], 40.0)

    def test_invalid_value_raises_and_does_not_mutate(self):
        with self.assertRaises((TypeError, ValueError)):
            self.state.update_distances(front=float("nan"))
        self.assertAlmostEqual(self._distances()["front"], 10.0)

    def test_invalid_bool_value_raises(self):
        with self.assertRaises(TypeError):
            self.state.update_distances(front=True)

    def test_invalid_string_value_raises(self):
        with self.assertRaises(TypeError):
            self.state.update_distances(left="far")

    def test_invalid_inf_raises(self):
        with self.assertRaises(ValueError):
            self.state.update_distances(right=float("inf"))

    def test_all_four_update_together(self):
        self.state.update_distances(front=1.0, rear=2.0, left=3.0, right=4.0)
        d = self._distances()
        self.assertAlmostEqual(d["front"], 1.0)
        self.assertAlmostEqual(d["rear"], 2.0)
        self.assertAlmostEqual(d["left"], 3.0)
        self.assertAlmostEqual(d["right"], 4.0)


if __name__ == "__main__":
    unittest.main()
