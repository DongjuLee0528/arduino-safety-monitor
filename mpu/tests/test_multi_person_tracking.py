import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from mpu.alert_manager import AlertManager
from mpu.dashboard_state import DashboardState
from mpu.main import HelmetDetectionSystem, _EventSuppressor
from mpu.sender import Sender
from mpu.tracker import PersonTracker

MAC_BENCHMARK_DISCLAIMER = "Mac M3 Max results are not Arduino UNO Q performance."
FRAME_SHAPE = (480, 640, 3)


def _frame():
    return np.zeros(FRAME_SHAPE, dtype="uint8")


def _person(bbox):
    return {"bbox": bbox, "confidence": 0.9}


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _make_system(clock=None):
    system = HelmetDetectionSystem.__new__(HelmetDetectionSystem)
    system.camera = MagicMock()
    system.person_detector = MagicMock()
    system.helmet_classifier = MagicMock()
    system.sender = MagicMock()
    system.bridge_rpc = MagicMock()
    system.alert_manager = AlertManager(callback=system.on_no_helmet_alert)
    system.running = False
    system._connected = False
    system._warning_hardware_active = False
    system._last_violation_detected = False
    system._clear_violation_count = 0
    system._tracked_incident_seen = False
    system.dashboard = DashboardState()
    system._events = _EventSuppressor(system.dashboard, suppress_seconds=0.0)
    system._prev_bboxes = []
    system.person_tracker = PersonTracker(clock=clock) if clock is not None else PersonTracker()
    system._next_incident_seq = 1
    system._pending_alert_frame = None
    system._pending_alert_label = None
    system._pending_alert_confidence = None
    system._pending_alert_metadata = None
    return system


def _update(tracker, bboxes):
    return tracker.update(bboxes, frame_shape=FRAME_SHAPE)


class TestPersonTracker(unittest.TestCase):
    def test_one_person_retains_track_id(self):
        tracker = PersonTracker()
        first, _ = _update(tracker, [[0, 0, 100, 200]])
        second, _ = _update(tracker, [[4, 3, 100, 200]])
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_detection_order_changes_identity_stays_stable(self):
        tracker = PersonTracker()
        tracks, _ = _update(tracker, [[0, 0, 100, 200], [400, 0, 100, 200]])
        left_id, right_id = tracks[0].track_id, tracks[1].track_id
        tracks, _ = _update(tracker, [[402, 0, 100, 200], [2, 0, 100, 200]])
        ids_by_x = {track.bbox[0]: track.track_id for track in tracks}
        self.assertEqual(ids_by_x[2], left_id)
        self.assertEqual(ids_by_x[402], right_id)

    def test_two_separated_people_receive_different_ids(self):
        tracker = PersonTracker()
        tracks, _ = _update(tracker, [[0, 0, 100, 200], [400, 0, 100, 200]])
        self.assertEqual(len({track.track_id for track in tracks}), 2)

    def test_detection_loss_0_5_seconds_preserves_track(self):
        clock = _Clock()
        tracker = PersonTracker(clock=clock)
        tracks, _ = _update(tracker, [[0, 0, 100, 200]])
        track_id = tracks[0].track_id
        clock.advance(0.5)
        _update(tracker, [])
        tracks, _ = _update(tracker, [[2, 0, 100, 200]])
        self.assertEqual(tracks[0].track_id, track_id)

    def test_detection_loss_1_5_seconds_preserves_track(self):
        clock = _Clock()
        tracker = PersonTracker(clock=clock)
        tracks, _ = _update(tracker, [[0, 0, 100, 200]])
        track_id = tracks[0].track_id
        clock.advance(1.5)
        _update(tracker, [])
        tracks, _ = _update(tracker, [[2, 0, 100, 200]])
        self.assertEqual(tracks[0].track_id, track_id)

    def test_disappearance_beyond_ttl_expires_and_prunes(self):
        clock = _Clock()
        tracker = PersonTracker(ttl_seconds=2.0, clock=clock)
        tracks, _ = _update(tracker, [[0, 0, 100, 200]])
        track_id = tracks[0].track_id
        clock.advance(2.1)
        _tracks, expired = _update(tracker, [])
        self.assertEqual([track.track_id for track in expired], [track_id])
        self.assertEqual(tracker.active_tracks(), [])

    def test_tracking_ttl_is_seconds_not_frames(self):
        for fps in (5, 10, 15, 30):
            with self.subTest(fps=fps):
                clock = _Clock()
                tracker = PersonTracker(clock=clock)
                tracks, _ = _update(tracker, [[0, 0, 100, 200]])
                track_id = tracks[0].track_id
                for _ in range(int(1.5 * fps)):
                    clock.advance(1.0 / fps)
                    _update(tracker, [])
                tracks, _ = _update(tracker, [[2, 0, 100, 200]])
                self.assertEqual(tracks[0].track_id, track_id)

    def test_track_storage_remains_bounded(self):
        tracker = PersonTracker(max_tracks=2, centroid_distance_ratio=0.001)
        _update(tracker, [[0, 0, 10, 10], [100, 0, 10, 10], [200, 0, 10, 10]])
        self.assertLessEqual(len(tracker.active_tracks()), 2)

    def test_unmatched_detection_receives_new_id(self):
        tracker = PersonTracker(centroid_distance_ratio=0.025)
        first, _ = _update(tracker, [[0, 0, 100, 200]])
        second, _ = _update(tracker, [[400, 0, 100, 200]])
        self.assertNotEqual(first[0].track_id, second[0].track_id)

    def test_ambiguous_matching_is_deterministic(self):
        tracker = PersonTracker(iou_threshold=0.0, centroid_distance_ratio=0.625)
        _update(tracker, [[0, 0, 100, 100], [120, 0, 100, 100]])
        first = [(track.track_id, track.bbox) for track in _update(tracker, [[60, 0, 100, 100], [180, 0, 100, 100]])[0]]
        tracker = PersonTracker(iou_threshold=0.0, centroid_distance_ratio=0.625)
        _update(tracker, [[0, 0, 100, 100], [120, 0, 100, 100]])
        second = [(track.track_id, track.bbox) for track in _update(tracker, [[60, 0, 100, 100], [180, 0, 100, 100]])[0]]
        self.assertEqual(first, second)

    def test_two_people_briefly_cross_without_extra_track_ids(self):
        tracker = PersonTracker()
        _update(tracker, [[0, 0, 100, 200], [260, 0, 100, 200]])
        _update(tracker, [[80, 0, 100, 200], [180, 0, 100, 200]])
        _update(tracker, [[160, 0, 100, 200], [100, 0, 100, 200]])
        self.assertLessEqual(max(track.track_id for track in tracker.active_tracks()), 2)


class TestPerTrackIncidents(unittest.TestCase):
    def test_track_a_creates_one_incident_and_continuation_does_not_duplicate(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(8):
            system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["warnings"], 1)
        system.sender.send_alert.assert_called_once()
        system.bridge_rpc.led_control.assert_called_once_with("red")

    def test_track_b_independently_creates_one_incident(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([400, 0, 100, 200]),
        ]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["warnings"], 2)
        self.assertEqual(system.sender.send_alert.call_count, 2)
        system.bridge_rpc.led_control.assert_called_once_with("red")

    def test_track_a_resolution_does_not_resolve_track_b(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([400, 0, 100, 200]),
        ]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        for _ in range(2):
            system.helmet_classifier.predict.side_effect = [
                {"label": "helmet", "confidence": 0.95},
                {"label": "no_helmet", "confidence": 0.9},
            ]
            system.process_frame(_frame())
        active = {track.track_id: track.incident_active for track in system.person_tracker.active_tracks()}
        self.assertFalse(active[1])
        self.assertTrue(active[2])
        self.assertNotIn(call("off"), system.bridge_rpc.led_control.mock_calls)

    def test_unknown_and_detection_failure_do_not_resolve_active_incident(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        system.helmet_classifier.predict.return_value = {"label": "unknown", "confidence": 0.0}
        system.process_frame(_frame())
        system.person_detector.detect.side_effect = RuntimeError("detector unavailable")
        system.process_frame(_frame())
        self.assertTrue(system.person_tracker.active_tracks()[0].incident_active)
        self.assertNotIn(call("off"), system.bridge_rpc.led_control.mock_calls)

    def test_missed_active_track_plus_other_helmet_does_not_clear_warning(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())

        system.person_detector.detect.return_value = [_person([400, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        system.process_frame(_frame())

        active = {track.track_id: track.incident_active for track in system.person_tracker.active_tracks()}
        self.assertTrue(active[1])
        self.assertEqual(system.bridge_rpc.led_control.mock_calls, [call("red")])

    def test_two_explicit_helmet_frames_resolve_only_corresponding_track(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        self.assertTrue(system.person_tracker.active_tracks()[0].incident_active)
        system.process_frame(_frame())
        self.assertFalse(system.person_tracker.active_tracks()[0].incident_active)

    def test_after_resolution_later_violation_creates_new_incident(self):
        system = _make_system()
        system.alert_manager.cooldown = 0.0
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        system.process_frame(_frame())
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["warnings"], 2)
        self.assertEqual(system.sender.send_alert.call_count, 2)

    def test_cooldown_expiration_cannot_retrigger_active_incident(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        with patch("mpu.main.time.time", return_value=100.0):
            for _ in range(3):
                system.process_frame(_frame())
        with patch("mpu.main.time.time", return_value=1000.0):
            for _ in range(3):
                system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["warnings"], 1)
        system.sender.send_alert.assert_called_once()

    def test_expired_active_track_is_not_safely_resolved(self):
        clock = _Clock()
        system = _make_system(clock=clock)
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        clock.advance(2.1)
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        messages = [event["message"] for event in system.dashboard.snapshot()["events"]]
        self.assertTrue(any("expired with lost track" in message for message in messages))
        self.assertNotIn(call("off"), system.bridge_rpc.led_control.mock_calls)

    def test_reacquisition_within_ttl_preserves_incident_and_side_effect_flags(self):
        clock = _Clock()
        system = _make_system(clock=clock)
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        track = system.person_tracker.active_tracks()[0]
        track_id = track.track_id
        incident_id = track.current_incident_id

        clock.advance(1.5)
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [_person([2, 0, 100, 200])]
        system.process_frame(_frame())

        track = system.person_tracker.active_tracks()[0]
        self.assertEqual(track.track_id, track_id)
        self.assertEqual(track.current_incident_id, incident_id)
        self.assertTrue(track.incident_active)
        self.assertTrue(track.http_event_sent)
        self.assertTrue(track.incident_counted)
        self.assertTrue(track.buzzer_requested)
        self.assertEqual(system.dashboard.snapshot()["statistics"]["warnings"], 1)
        system.sender.send_alert.assert_called_once()
        system.bridge_rpc.led_control.assert_called_once_with("red")

    def test_missing_observation_cannot_clear_active_incident(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        system.person_detector.detect.return_value = []
        system.process_frame(_frame())
        system.person_detector.detect.return_value = [_person([2, 0, 100, 200])]
        system.process_frame(_frame())
        self.assertTrue(system.person_tracker.active_tracks()[0].incident_active)
        self.assertNotIn(call("off"), system.bridge_rpc.led_control.mock_calls)


class TestSideEffectsAndHttp(unittest.TestCase):
    def test_transport_retry_reuses_same_incident_metadata(self):
        payloads = []

        class Session:
            def post(self, _url, json=None, timeout=None):
                payloads.append(json)
                response = MagicMock()
                response.status_code = 500 if len(payloads) == 1 else 200
                response.text = "retry"
                return response

        sender = Sender.__new__(Sender)
        sender.server_url = "http://example.invalid/api/alert"
        sender.session = Session()
        with patch("mpu.sender.time.sleep", return_value=None):
            sender.send_alert(
                _frame(),
                "no_helmet",
                0.9,
                retries=1,
                metadata={"track_id": 1, "incident_id": "helmet-1-1"},
            )
        self.assertEqual(payloads[0]["metadata"], payloads[1]["metadata"])

    def test_second_simultaneous_incident_coalesces_buzzer_but_keeps_count_and_http(self):
        system = _make_system()
        system.person_detector.detect.return_value = [
            _person([0, 0, 100, 200]),
            _person([400, 0, 100, 200]),
        ]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        self.assertEqual(system.dashboard.snapshot()["statistics"]["warnings"], 2)
        self.assertEqual(system.sender.send_alert.call_count, 2)
        system.bridge_rpc.led_control.assert_called_once_with("red")
        messages = [event["message"] for event in system.dashboard.snapshot()["events"]]
        self.assertIn("No-helmet warning buzzer request coalesced", messages)

    def test_continuous_incident_does_not_restart_buzzer_after_physical_stop(self):
        system = _make_system()
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        system._warning_hardware_active = False
        for _ in range(5):
            system.process_frame(_frame())
        system.bridge_rpc.led_control.assert_called_once_with("red")

    def test_genuine_resolved_reactivated_incident_requests_new_buzzer(self):
        system = _make_system()
        system.alert_manager.cooldown = 0.0
        system.person_detector.detect.return_value = [_person([0, 0, 100, 200])]
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        system.helmet_classifier.predict.return_value = {"label": "helmet", "confidence": 0.95}
        system.process_frame(_frame())
        system.process_frame(_frame())
        system.helmet_classifier.predict.return_value = {"label": "no_helmet", "confidence": 0.9}
        for _ in range(3):
            system.process_frame(_frame())
        self.assertEqual(system.bridge_rpc.led_control.mock_calls, [call("red"), call("off"), call("red")])


class TestTrackingPerformance(unittest.TestCase):
    def test_tracker_benchmark_disclaimer_is_persisted(self):
        self.assertIn("not Arduino UNO Q performance", MAC_BENCHMARK_DISCLAIMER)

    def test_tracker_long_sequence_does_not_grow_unbounded(self):
        tracker = PersonTracker(max_tracks=4, centroid_distance_ratio=0.001)
        for frame_index in range(200):
            _update(tracker, [[frame_index * 20, 0, 10, 10]])
        self.assertLessEqual(len(tracker.active_tracks()), 4)
