"""
Tests for runtime event logging wired into HelmetDetectionSystem.

Covers:
  - Connection events (connect / disconnect)
  - Detection events (helmet, no-helmet, worker)
  - Alert event
  - Duplicate suppression
  - Queue size respected
  - Snapshot contains events
  - No regression of existing pipeline
  - Disconnect transition semantics (blocking fix 1)
  - Distinct-worker event suppression (blocking fix 2)
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from mpu.dashboard_state import DashboardState, EventType
from mpu.main import HelmetDetectionSystem, _EventSuppressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system():
    system = HelmetDetectionSystem.__new__(HelmetDetectionSystem)
    system.camera = MagicMock()
    system.person_detector = MagicMock()
    system.helmet_classifier = MagicMock()
    system.sender = MagicMock()
    system.bridge_rpc = MagicMock()
    system.alert_manager = MagicMock()
    system.running = False
    system.alert_hardware_active = False
    system._connected = False
    system.dashboard = DashboardState()
    system._events = _EventSuppressor(system.dashboard, suppress_seconds=0.0)
    system._prev_bboxes = []
    return system


def _frame():
    return np.zeros((480, 640, 3), dtype="uint8")


def _events(system):
    return system.dashboard.snapshot()["events"]


def _event_messages(system):
    return [e["message"] for e in _events(system)]


def _event_types(system):
    return [e["event_type"] for e in _events(system)]


def _count_message(system, msg):
    return sum(1 for m in _event_messages(system) if m == msg)


# ---------------------------------------------------------------------------
# _EventSuppressor unit tests
# ---------------------------------------------------------------------------

class TestEventSuppressor(unittest.TestCase):
    def test_first_call_emits(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=10.0)
        result = sup.append(EventType.SYSTEM, "hello")
        self.assertTrue(result)
        self.assertEqual(len(ds.snapshot()["events"]), 1)

    def test_duplicate_within_window_suppressed(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=10.0)
        sup.append(EventType.SYSTEM, "hello")
        result = sup.append(EventType.SYSTEM, "hello")
        self.assertFalse(result)
        self.assertEqual(len(ds.snapshot()["events"]), 1)

    def test_different_messages_both_emitted(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=10.0)
        sup.append(EventType.SYSTEM, "msg-a")
        sup.append(EventType.SYSTEM, "msg-b")
        self.assertEqual(len(ds.snapshot()["events"]), 2)

    def test_zero_suppress_always_emits(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=0.0)
        sup.append(EventType.SYSTEM, "x")
        sup.append(EventType.SYSTEM, "x")
        self.assertEqual(len(ds.snapshot()["events"]), 2)

    def test_event_type_stored_correctly(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=0.0)
        sup.append(EventType.CONNECTION, "Robot connected")
        event = ds.snapshot()["events"][0]
        self.assertEqual(event["event_type"], "connection")

    def test_message_stored_correctly(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=0.0)
        sup.append(EventType.DETECTION, "Worker detected")
        self.assertEqual(ds.snapshot()["events"][0]["message"], "Worker detected")

    def test_timestamp_present_in_event(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=0.0)
        sup.append(EventType.DETECTION, "Worker detected")
        self.assertIn("timestamp", ds.snapshot()["events"][0])

    def test_explicit_key_same_message_different_keys_both_emitted(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=10.0)
        sup.append(EventType.DETECTION, "Worker detected", key="worker:0:0:100:200:detected")
        sup.append(EventType.DETECTION, "Worker detected", key="worker:400:0:100:200:detected")
        self.assertEqual(len(ds.snapshot()["events"]), 2)

    def test_explicit_key_same_key_suppressed(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=10.0)
        sup.append(EventType.DETECTION, "Worker detected", key="worker:0:0:100:200:detected")
        result = sup.append(EventType.DETECTION, "Worker detected", key="worker:0:0:100:200:detected")
        self.assertFalse(result)
        self.assertEqual(len(ds.snapshot()["events"]), 1)

    def test_key_none_falls_back_to_message(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=10.0)
        sup.append(EventType.CONNECTION, "Robot connected", key=None)
        result = sup.append(EventType.CONNECTION, "Robot connected", key=None)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Disconnect transition (blocking fix 1)
# ---------------------------------------------------------------------------

class TestDisconnectTransition(unittest.TestCase):
    def test_stop_on_never_connected_emits_no_disconnect_event(self):
        system = _make_system()
        system.stop()
        self.assertNotIn("Robot disconnected", _event_messages(system))

    def test_successful_connection_then_stop_emits_exactly_one_disconnect(self):
        system = _make_system()
        system.bridge_rpc.connect.return_value = None
        system.camera.capture_frame.side_effect = RuntimeError("stop")
        system.person_detector.detect.return_value = []
        system.start()
        self.assertEqual(_count_message(system, "Robot disconnected"), 1)

    def test_repeated_stop_does_not_emit_duplicate_disconnect(self):
        system = _make_system()
        system._connected = True
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=0.0)
        system.stop()
        system.stop()
        self.assertEqual(_count_message(system, "Robot disconnected"), 1)

    def test_dashboard_connection_offline_after_real_disconnect(self):
        system = _make_system()
        system._connected = True
        system.stop()
        self.assertEqual(system.dashboard.snapshot()["connection"]["status"], "offline")

    def test_failed_connection_then_stop_emits_no_disconnect_event(self):
        system = _make_system()
        system.bridge_rpc.connect.side_effect = Exception("port unavailable")
        try:
            system.start()
        except Exception:
            pass
        self.assertNotIn("Robot disconnected", _event_messages(system))

    def test_connected_flag_false_initially(self):
        system = _make_system()
        self.assertFalse(system._connected)

    def test_connected_flag_true_after_successful_connect(self):
        system = _make_system()
        system.bridge_rpc.connect.return_value = None
        system.camera.capture_frame.side_effect = RuntimeError("stop")
        system.person_detector.detect.return_value = []

        recorded_connected = []
        orig_stop = system.stop
        def _spy_stop():
            recorded_connected.append(system._connected)
            orig_stop()
        system.stop = _spy_stop

        system.start()
        # _connected must have been True at the point stop() ran (before it clears it).
        self.assertTrue(recorded_connected[0])


# ---------------------------------------------------------------------------
# Distinct-worker event suppression (blocking fix 2)
# ---------------------------------------------------------------------------

class TestDistinctWorkerEvents(unittest.TestCase):
    def _two_worker_frame(self, system, label_a="helmet", label_b="helmet"):
        system.person_detector.detect.return_value = [
            {"bbox": [0, 0, 100, 200], "confidence": 0.9},
            {"bbox": [400, 0, 100, 200], "confidence": 0.9},
        ]
        system.helmet_classifier.predict.side_effect = [
            {"label": label_a, "confidence": 0.9},
            {"label": label_b, "confidence": 0.9},
        ]
        system.process_frame(_frame())

    def test_two_distinct_workers_produce_two_worker_detected_events(self):
        system = _make_system()
        self._two_worker_frame(system)
        self.assertEqual(_count_message(system, "Worker detected"), 2)

    def test_two_distinct_helmet_workers_produce_two_helmet_events(self):
        system = _make_system()
        self._two_worker_frame(system, "helmet", "helmet")
        helmet_evs = [e for e in _events(system) if "Helmet detected" in e["message"]]
        self.assertEqual(len(helmet_evs), 2)

    def test_two_distinct_no_helmet_workers_produce_two_no_helmet_events(self):
        system = _make_system()
        self._two_worker_frame(system, "no_helmet", "no_helmet")
        no_helmet_evs = [e for e in _events(system) if "No helmet detected" in e["message"]]
        self.assertEqual(len(no_helmet_evs), 2)

    def test_same_worker_repeated_across_frames_does_not_flood_events(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)
        bbox = [0, 0, 100, 200]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        for _ in range(60):
            system.person_detector.detect.return_value = [{"bbox": bbox, "confidence": 0.9}]
            system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
            system.process_frame(_frame())
        self.assertEqual(_count_message(system, "Worker detected"), 1)

    def test_true_duplicate_command_error_remains_suppressed(self):
        ds = DashboardState()
        sup = _EventSuppressor(ds, suppress_seconds=10.0)
        sup.append(EventType.SYSTEM, "Arduino communication failed: timeout")
        result = sup.append(EventType.SYSTEM, "Arduino communication failed: timeout")
        self.assertFalse(result)
        self.assertEqual(len(ds.snapshot()["events"]), 1)

    def test_event_messages_are_human_readable(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        for ev in _events(system):
            self.assertNotIn("worker:", ev["message"])

    def test_queue_bounded_behavior_with_many_workers(self):
        ds = DashboardState(max_events=5)
        system = HelmetDetectionSystem.__new__(HelmetDetectionSystem)
        system.camera = MagicMock()
        system.person_detector = MagicMock()
        system.helmet_classifier = MagicMock()
        system.sender = MagicMock()
        system.bridge_rpc = MagicMock()
        system.alert_manager = MagicMock()
        system.running = False
        system.alert_hardware_active = False
        system._connected = False
        system.dashboard = ds
        system._events = _EventSuppressor(ds, suppress_seconds=0.0)
        system._prev_bboxes = []

        for i in range(10):
            system.person_detector.detect.return_value = [
                {"bbox": [i * 10, 0, 50, 100], "confidence": 0.9}
            ]
            system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
            system.process_frame(_frame())
            system._prev_bboxes = []

        self.assertLessEqual(len(ds.snapshot()["events"]), 5)


# ---------------------------------------------------------------------------
# Accepted-worker event guarantee (blocking fix 3)
# ---------------------------------------------------------------------------

class TestAcceptedWorkerEventGuarantee(unittest.TestCase):
    def test_accepted_worker_always_generates_worker_detected(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        self.assertEqual(_count_message(system, "Worker detected"), 1)

    def test_accepted_helmet_worker_always_generates_helmet_detected(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        self.assertTrue(any("Helmet detected" in m for m in _event_messages(system)))

    def test_accepted_no_helmet_worker_always_generates_no_helmet_detected(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.88}
        system.process_frame(_frame())
        self.assertTrue(any("No helmet detected" in m for m in _event_messages(system)))

    def test_leave_and_reenter_generates_new_events(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)
        bbox = [0, 0, 100, 200]
        system.person_detector.detect.return_value = [{"bbox": bbox, "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [{"bbox": bbox, "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        self.assertEqual(_count_message(system, "Worker detected"), 2)

    def test_system_events_remain_suppressible(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)
        system._events.append(EventType.SYSTEM, "Arduino communication failed: timeout")
        result = system._events.append(EventType.SYSTEM, "Arduino communication failed: timeout")
        self.assertFalse(result)
        self.assertEqual(_count_message(system, "Arduino communication failed: timeout"), 1)

    def test_same_frame_rejected_detections_generate_no_events(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)
        bbox = [0, 0, 100, 200]
        system.person_detector.detect.return_value = [{"bbox": bbox, "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        count_after_first = _count_message(system, "Worker detected")
        system.person_detector.detect.return_value = [{"bbox": bbox, "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        self.assertEqual(_count_message(system, "Worker detected"), count_after_first)


# ---------------------------------------------------------------------------
# Connection events
# ---------------------------------------------------------------------------

class TestConnectionEvents(unittest.TestCase):
    def test_connect_logs_connected_event(self):
        system = _make_system()
        system.bridge_rpc.connect.return_value = None
        system.camera.capture_frame.side_effect = RuntimeError("stop")
        system.person_detector.detect.return_value = []

        recorded = []
        real_append = system._events.append
        def _spy(et, msg, key=None):
            recorded.append((et, msg))
            return real_append(et, msg, key=key)
        system._events.append = _spy

        system.start()

        connection_msgs = [msg for et, msg in recorded if msg == "Robot connected"]
        self.assertTrue(len(connection_msgs) >= 1)

    def test_connect_event_type_is_connection(self):
        system = _make_system()
        system.bridge_rpc.connect.return_value = None
        system.camera.capture_frame.side_effect = RuntimeError("stop")
        system.person_detector.detect.return_value = []

        recorded = []
        real_append = system._events.append
        def _spy(et, msg, key=None):
            recorded.append((et, msg))
            return real_append(et, msg, key=key)
        system._events.append = _spy

        system.start()

        for et, msg in recorded:
            if msg == "Robot connected":
                self.assertEqual(et, EventType.CONNECTION)
                return
        self.fail("Robot connected event not found")


# ---------------------------------------------------------------------------
# Detection events
# ---------------------------------------------------------------------------

class TestDetectionEvents(unittest.TestCase):
    def test_helmet_worker_logs_helmet_event(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}

        system.process_frame(_frame())

        messages = _event_messages(system)
        self.assertTrue(any("Helmet detected" in m for m in messages))

    def test_no_helmet_worker_logs_no_helmet_event(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.88}

        system.process_frame(_frame())

        messages = _event_messages(system)
        self.assertTrue(any("No helmet detected" in m for m in messages))

    def test_new_worker_logs_worker_detected_event(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}

        system.process_frame(_frame())

        self.assertIn("Worker detected", _event_messages(system))

    def test_detection_event_type_correct(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}

        system.process_frame(_frame())

        for ev in _events(system):
            if "Helmet detected" in ev["message"]:
                self.assertEqual(ev["event_type"], "detection")
                return
        self.fail("Helmet detected event not found")

    def test_confidence_included_in_detection_message(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}

        system.process_frame(_frame())

        messages = _event_messages(system)
        self.assertTrue(any("0.95" in m for m in messages))

    def test_no_person_no_detection_event(self):
        system = _make_system()
        system.person_detector.detect.return_value = []

        system.process_frame(_frame())

        messages = _event_messages(system)
        self.assertFalse(any("detected" in m.lower() for m in messages))

    def test_same_worker_many_frames_worker_event_emitted_once(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)

        for _ in range(120):
            system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
            system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
            system.process_frame(_frame())

        worker_events = [e for e in _events(system) if e["message"] == "Worker detected"]
        self.assertEqual(len(worker_events), 1)


# ---------------------------------------------------------------------------
# Alert event
# ---------------------------------------------------------------------------

class TestAlertEvent(unittest.TestCase):
    def test_no_helmet_alert_logs_alert_event(self):
        system = _make_system()
        system.bridge_rpc.led_control.return_value = True
        system.bridge_rpc.buzzer_control.return_value = True

        system.on_no_helmet_alert()

        messages = _event_messages(system)
        self.assertIn("No-helmet alert triggered", messages)

    def test_alert_event_type_is_alert(self):
        system = _make_system()
        system.bridge_rpc.led_control.return_value = True
        system.bridge_rpc.buzzer_control.return_value = True

        system.on_no_helmet_alert()

        for ev in _events(system):
            if ev["message"] == "No-helmet alert triggered":
                self.assertEqual(ev["event_type"], "alert")
                return
        self.fail("Alert event not found")

    def test_failed_alert_command_does_not_log_alert_event(self):
        system = _make_system()
        system.bridge_rpc.led_control.side_effect = Exception("disconnected")

        system.on_no_helmet_alert()

        messages = _event_messages(system)
        self.assertNotIn("No-helmet alert triggered", messages)


# ---------------------------------------------------------------------------
# Duplicate suppression
# ---------------------------------------------------------------------------

class TestDuplicateSuppression(unittest.TestCase):
    def test_duplicate_within_window_not_logged(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)

        system._events.append(EventType.CONNECTION, "Robot connected")
        system._events.append(EventType.CONNECTION, "Robot connected")

        events = [e for e in _events(system) if e["message"] == "Robot connected"]
        self.assertEqual(len(events), 1)

    def test_different_messages_both_logged(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)

        system._events.append(EventType.CONNECTION, "Robot connected")
        system._events.append(EventType.CONNECTION, "Robot disconnected")

        self.assertEqual(len(_events(system)), 2)

    def test_suppressor_returns_false_on_duplicate(self):
        system = _make_system()
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=10.0)

        system._events.append(EventType.SYSTEM, "err")
        result = system._events.append(EventType.SYSTEM, "err")
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Queue size
# ---------------------------------------------------------------------------

class TestQueueSize(unittest.TestCase):
    def test_events_bounded_by_max_events(self):
        ds = DashboardState(max_events=5)
        sup = _EventSuppressor(ds, suppress_seconds=0.0)
        for i in range(10):
            sup.append(EventType.SYSTEM, f"msg-{i}")
        self.assertLessEqual(len(ds.snapshot()["events"]), 5)

    def test_oldest_event_dropped_when_full(self):
        ds = DashboardState(max_events=3)
        sup = _EventSuppressor(ds, suppress_seconds=0.0)
        for i in range(5):
            sup.append(EventType.SYSTEM, f"msg-{i}")
        messages = [e["message"] for e in ds.snapshot()["events"]]
        self.assertNotIn("msg-0", messages)
        self.assertIn("msg-4", messages)


# ---------------------------------------------------------------------------
# Snapshot contains events
# ---------------------------------------------------------------------------

class TestSnapshotContainsEvents(unittest.TestCase):
    def test_snapshot_events_key_present(self):
        system = _make_system()
        snap = system.dashboard.snapshot()
        self.assertIn("events", snap)

    def test_snapshot_events_after_detection(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.9}
        system.process_frame(_frame())
        snap = system.dashboard.snapshot()
        self.assertGreater(len(snap["events"]), 0)

    def test_snapshot_event_has_required_keys(self):
        system = _make_system()
        system._events.append(EventType.CONNECTION, "Robot connected")
        event = system.dashboard.snapshot()["events"][0]
        self.assertIn("timestamp", event)
        self.assertIn("event_type", event)
        self.assertIn("message", event)

    def test_snapshot_events_list_is_independent_copy(self):
        system = _make_system()
        system._events.append(EventType.CONNECTION, "Robot connected")
        snap1 = system.dashboard.snapshot()
        snap1["events"].clear()
        snap2 = system.dashboard.snapshot()
        self.assertGreater(len(snap2["events"]), 0)


# ---------------------------------------------------------------------------
# No regression of existing pipeline
# ---------------------------------------------------------------------------

class TestNoRegressionPipeline(unittest.TestCase):
    def test_alert_manager_still_called(self):
        system = _make_system()
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.alert_manager.on_detection.assert_called_once_with(False)

    def test_sender_called_on_no_helmet(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.88}
        system.process_frame(_frame())
        system.sender.send_alert.assert_called_once()

    def test_sender_not_called_on_helmet(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        system.sender.send_alert.assert_not_called()

    def test_dashboard_detection_still_updated(self):
        system = _make_system()
        system.person_detector.detect.return_value = [{"bbox": [0, 0, 100, 200], "confidence": 0.9}]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.88}
        system.process_frame(_frame())
        snap = system.dashboard.snapshot()["detection"]
        self.assertTrue(snap["worker_present"])
        self.assertEqual(snap["helmet_result"], "no_helmet")

    def test_stop_sets_offline(self):
        system = _make_system()
        system.dashboard.update_connection(online=True)
        system.stop()
        self.assertEqual(system.dashboard.snapshot()["connection"]["status"], "offline")


if __name__ == "__main__":
    unittest.main()
