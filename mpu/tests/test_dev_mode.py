"""
Tests for Hardware-Free Development Mode (APP_LAB_DEV_MODE).

Coverage:
  A – Strict mode (dev_mode=False): camera init failure → RuntimeError propagated
  B – Dev mode (dev_mode=True): camera init failure → WARNING logged, system continues
  C – Strict mode does not swallow camera errors (no silent pass)
  D – stop_capture() cv2.destroyAllWindows() headless: cv2.error caught, no exception raised
  E – stop_capture() cv2.destroyAllWindows() success case: called exactly once, no exception
  F – APP_LAB_DEV_MODE boolean parsing: true/1/yes → True; false/0/no/empty → False
  G – HelmetDetectionSystem.start() in dev mode: exits immediately, logs warning, no hardware I/O
  H – Regression: existing strict-mode camera init still raises on genuine failure
  I – CameraCapture cleanup idempotency: release/destroyAllWindows at most once per lifecycle
  J – CameraCapture destructor: no-op after explicit stop_capture(); best-effort without it
  K – HelmetDetectionSystem.stop() cleanup ownership: no duplicate destroyAllWindows
"""

import importlib
import logging
import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch

import cv2

from mpu.camera import CameraCapture
from mpu.dashboard_state import DashboardState, EventType
from mpu.main import HelmetDetectionSystem, _EventSuppressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system_dev(dev_mode: bool = True) -> HelmetDetectionSystem:
    """Construct a HelmetDetectionSystem with all I/O mocked; honour dev_mode."""
    system = HelmetDetectionSystem.__new__(HelmetDetectionSystem)
    system.camera = MagicMock()
    system.camera._camera_available = not dev_mode
    system.person_detector = MagicMock()
    system.helmet_classifier = MagicMock()
    system.sender = MagicMock()
    system.bridge_rpc = MagicMock()
    system.alert_manager = MagicMock()
    system.running = False
    system.alert_hardware_active = False
    system._connected = False
    system._dev_mode = dev_mode
    system.dashboard = DashboardState()
    system._events = _EventSuppressor(system.dashboard, suppress_seconds=0.0)
    system._prev_bboxes = []
    system._last_tick_time = 0.0
    return system


# ---------------------------------------------------------------------------
# A – Strict mode: RuntimeError from camera propagates
# ---------------------------------------------------------------------------

class TestStrictModeCameraFailure(unittest.TestCase):
    def test_A_strict_camera_init_raises_runtime_error(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            with self.assertRaises(RuntimeError) as ctx:
                CameraCapture(camera_index=99, dev_mode=False)

        self.assertIn("99", str(ctx.exception))


# ---------------------------------------------------------------------------
# B – Dev mode: RuntimeError from camera → WARNING, no exception
# ---------------------------------------------------------------------------

class TestDevModeCameraFailure(unittest.TestCase):
    def test_B_dev_mode_camera_init_failure_logs_warning_not_raises(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            with self.assertLogs("mpu.camera", level="WARNING") as log_ctx:
                cam = CameraCapture(camera_index=0, dev_mode=True)

        self.assertFalse(cam._camera_available)
        self.assertTrue(any("DEV MODE" in msg for msg in log_ctx.output))

    def test_B_dev_mode_camera_available_false_when_init_fails(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=True)

        self.assertFalse(cam._camera_available)

    def test_B_dev_mode_capture_frame_raises_when_unavailable(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=True)

        with self.assertRaises(RuntimeError):
            cam.capture_frame()


# ---------------------------------------------------------------------------
# C – Strict mode does not silently pass camera errors
# ---------------------------------------------------------------------------

class TestStrictModeNoSilentSwallow(unittest.TestCase):
    def test_C_strict_mode_not_dev_mode_flag(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            raised = False
            try:
                CameraCapture(camera_index=0, dev_mode=False)
            except RuntimeError:
                raised = True

        self.assertTrue(raised, "Strict mode must raise RuntimeError on camera failure")


# ---------------------------------------------------------------------------
# D – Headless destroyAllWindows: cv2.error caught, cleanup completes
# ---------------------------------------------------------------------------

class TestHeadlessDestroyAllWindows(unittest.TestCase):
    def test_D_stop_capture_survives_cv2_error_from_destroy_all_windows(self):
        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=cv2.error("headless")):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            try:
                cam.stop_capture()
            except cv2.error:
                self.fail("stop_capture() must not propagate cv2.error from destroyAllWindows()")

    def test_D_stop_capture_releases_cap_before_destroy(self):
        release_calls = []
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.release.side_effect = lambda: release_calls.append(1)
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            cam.stop_capture()

        self.assertEqual(len(release_calls), 1, "cap.release() must be called once")
        self.assertEqual(len(destroy_calls), 1, "destroyAllWindows must be called once")


# ---------------------------------------------------------------------------
# E – Normal (non-headless) destroyAllWindows: called exactly once, no exception
# ---------------------------------------------------------------------------

class TestNormalDestroyAllWindows(unittest.TestCase):
    def test_E_stop_capture_calls_destroy_all_windows_once(self):
        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows") as mock_destroy:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            cam.stop_capture()

        mock_destroy.assert_called_once()


# ---------------------------------------------------------------------------
# F – APP_LAB_DEV_MODE boolean parsing
# ---------------------------------------------------------------------------

class TestDevModeBooleanParsing(unittest.TestCase):
    def _reload_config(self, env_value: str) -> types.ModuleType:
        with patch.dict("os.environ", {"APP_LAB_DEV_MODE": env_value}, clear=False):
            if "mpu.config" in sys.modules:
                del sys.modules["mpu.config"]
            mod = importlib.import_module("mpu.config")
        return mod

    def test_F_true_string_activates_dev_mode(self):
        mod = self._reload_config("true")
        self.assertTrue(mod.APP_LAB_DEV_MODE)

    def test_F_1_string_activates_dev_mode(self):
        mod = self._reload_config("1")
        self.assertTrue(mod.APP_LAB_DEV_MODE)

    def test_F_yes_string_activates_dev_mode(self):
        mod = self._reload_config("yes")
        self.assertTrue(mod.APP_LAB_DEV_MODE)

    def test_F_TRUE_uppercase_activates_dev_mode(self):
        mod = self._reload_config("TRUE")
        self.assertTrue(mod.APP_LAB_DEV_MODE)

    def test_F_false_string_is_strict(self):
        mod = self._reload_config("false")
        self.assertFalse(mod.APP_LAB_DEV_MODE)

    def test_F_0_string_is_strict(self):
        mod = self._reload_config("0")
        self.assertFalse(mod.APP_LAB_DEV_MODE)

    def test_F_no_string_is_strict(self):
        mod = self._reload_config("no")
        self.assertFalse(mod.APP_LAB_DEV_MODE)

    def test_F_empty_string_is_strict(self):
        mod = self._reload_config("")
        self.assertFalse(mod.APP_LAB_DEV_MODE)

    def tearDown(self):
        if "mpu.config" in sys.modules:
            del sys.modules["mpu.config"]


# ---------------------------------------------------------------------------
# G – HelmetDetectionSystem.start() in dev mode exits immediately
# ---------------------------------------------------------------------------

class TestHelmetSystemDevModeStart(unittest.TestCase):
    def test_G_start_in_dev_mode_does_not_call_bridge_rpc_connect(self):
        system = _make_system_dev(dev_mode=True)
        system.start()
        system.bridge_rpc.connect.assert_not_called()

    def test_G_start_in_dev_mode_does_not_call_camera_capture_frame(self):
        system = _make_system_dev(dev_mode=True)
        system.start()
        system.camera.capture_frame.assert_not_called()

    def test_G_start_in_dev_mode_logs_warning(self):
        system = _make_system_dev(dev_mode=True)
        with self.assertLogs("mpu.main", level="WARNING") as log_ctx:
            system.start()
        self.assertTrue(any("DEV MODE" in msg for msg in log_ctx.output))

    def test_G_start_in_dev_mode_sets_running_false_on_exit(self):
        system = _make_system_dev(dev_mode=True)
        system.start()
        self.assertFalse(system.running)

    def test_G_start_in_dev_mode_appends_system_event(self):
        system = _make_system_dev(dev_mode=True)
        system.start()
        snap = system.dashboard.snapshot()
        events = snap.get("events", [])
        self.assertTrue(
            any("DEV MODE" in e.get("message", "") for e in events),
            "Expected a DEV MODE event in the dashboard",
        )


# ---------------------------------------------------------------------------
# H – Regression: strict-mode camera init raises on genuine hardware failure
# ---------------------------------------------------------------------------

class TestRegressionStrictMode(unittest.TestCase):
    def test_H_strict_mode_still_raises_on_camera_failure(self):
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            with self.assertRaises(RuntimeError):
                CameraCapture(camera_index=0, dev_mode=False)

    def test_H_dev_mode_false_is_default_when_env_unset(self):
        import os
        env_backup = os.environ.pop("APP_LAB_DEV_MODE", None)
        try:
            if "mpu.config" in sys.modules:
                del sys.modules["mpu.config"]
            import mpu.config as cfg
            self.assertFalse(cfg.APP_LAB_DEV_MODE)
        finally:
            if env_backup is not None:
                os.environ["APP_LAB_DEV_MODE"] = env_backup
            if "mpu.config" in sys.modules:
                del sys.modules["mpu.config"]


# ---------------------------------------------------------------------------
# I – CameraCapture cleanup idempotency
# ---------------------------------------------------------------------------

class TestStopCaptureIdempotency(unittest.TestCase):
    """
    I: stop_capture() is idempotent — release() and destroyAllWindows() each
    occur at most once regardless of how many times stop_capture() is called,
    and regardless of whether __del__ also runs.
    """

    def _make_cam(self):
        """Return a CameraCapture with a real (mocked) opened device."""
        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows"):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap
            cam = CameraCapture(camera_index=0, dev_mode=False)
        cam._mock_cap = mock_cap
        return cam

    def test_I1_stop_once_release_once_destroy_once(self):
        release_calls = []
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.release.side_effect = lambda: release_calls.append(1)
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            cam.stop_capture()

        self.assertEqual(release_calls, [1], "cap.release() must be called exactly once")
        self.assertEqual(destroy_calls, [1], "destroyAllWindows must be called exactly once")

    def test_I2_stop_twice_release_once_destroy_once(self):
        release_calls = []
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.release.side_effect = lambda: release_calls.append(1)
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            cam.stop_capture()
            cam.stop_capture()

        self.assertEqual(release_calls, [1], "cap.release() must not be called twice")
        self.assertEqual(destroy_calls, [1], "destroyAllWindows must not be called twice")

    def test_I3_stop_many_times_still_at_most_once(self):
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            for _ in range(5):
                cam.stop_capture()

        self.assertEqual(len(destroy_calls), 1)

    def test_I4_cleanup_done_flag_set_after_stop(self):
        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows"):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            self.assertFalse(cam._cleanup_done)
            cam.stop_capture()
            self.assertTrue(cam._cleanup_done)

    def test_I5_release_before_destroy(self):
        call_order = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: call_order.append("destroy")):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.release.side_effect = lambda: call_order.append("release")
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            cam.stop_capture()

        self.assertEqual(call_order, ["release", "destroy"],
                         "cap.release() must happen before cv2.destroyAllWindows()")

    def test_I6_cv2_error_from_destroy_does_not_raise(self):
        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=cv2.error("headless")):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            try:
                cam.stop_capture()
            except cv2.error:
                self.fail("stop_capture() must not propagate cv2.error from destroyAllWindows()")

    def test_I7_release_raises_destroy_still_called(self):
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.release.side_effect = RuntimeError("usb gone")
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            cam.stop_capture()

        self.assertEqual(destroy_calls, [1],
                         "destroyAllWindows must still be called even when release() raises")


# ---------------------------------------------------------------------------
# J – CameraCapture destructor behavior
# ---------------------------------------------------------------------------

class TestCameraDestructor(unittest.TestCase):
    """
    J: __del__ is a no-op after explicit stop_capture(); runs best-effort cleanup
    when stop_capture() was never called.
    """

    def test_J1_del_after_stop_no_extra_release(self):
        release_calls = []
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.release.side_effect = lambda: release_calls.append(1)
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            cam.stop_capture()
            cam.__del__()

        self.assertEqual(release_calls, [1], "__del__ after stop_capture must not repeat release")
        self.assertEqual(destroy_calls, [1], "__del__ after stop_capture must not repeat destroyAllWindows")

    def test_J2_del_without_prior_stop_performs_cleanup(self):
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            cam.__del__()

        self.assertEqual(destroy_calls, [1],
                         "__del__ without prior stop must perform cleanup once")

    def test_J3_del_does_not_raise(self):
        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=cv2.error("headless")):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            cam = CameraCapture(camera_index=0, dev_mode=False)
            try:
                cam.__del__()
            except Exception as exc:
                self.fail(f"__del__ must never propagate an exception; got {exc!r}")


# ---------------------------------------------------------------------------
# K – HelmetDetectionSystem.stop() cleanup ownership
# ---------------------------------------------------------------------------

class TestHelmetSystemStopCleanupOwnership(unittest.TestCase):
    """
    K: HelmetDetectionSystem.stop() delegates cv2.destroyAllWindows() entirely
    to CameraCapture.stop_capture().  It must NOT make an additional independent
    call.  Repeated stop() remains safe.
    """

    def _make_system_with_real_camera_mock(self):
        """System whose camera is a real CameraCapture with mocked cv2."""
        system = HelmetDetectionSystem.__new__(HelmetDetectionSystem)
        system.person_detector = MagicMock()
        system.helmet_classifier = MagicMock()
        system.sender = MagicMock()
        system.bridge_rpc = MagicMock()
        system.alert_manager = MagicMock()
        system.running = False
        system.alert_hardware_active = False
        system._connected = False
        system._dev_mode = False
        system.dashboard = DashboardState()
        from mpu.main import _EventSuppressor
        system._events = _EventSuppressor(system.dashboard, suppress_seconds=0.0)
        system._prev_bboxes = []
        system._last_tick_time = 0.0
        return system

    def test_K1_stop_causes_exactly_one_destroyAllWindows_via_camera(self):
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            system = self._make_system_with_real_camera_mock()
            system.camera = CameraCapture(camera_index=0, dev_mode=False)
            system.stop()

        self.assertEqual(destroy_calls, [1],
                         "Exactly one destroyAllWindows call expected across the full stop() sequence")

    def test_K2_stop_twice_still_one_destroyAllWindows(self):
        destroy_calls = []

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.destroyAllWindows", side_effect=lambda: destroy_calls.append(1)):
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap

            system = self._make_system_with_real_camera_mock()
            system.camera = CameraCapture(camera_index=0, dev_mode=False)
            system.stop()
            system.stop()

        self.assertEqual(destroy_calls, [1],
                         "Repeated stop() must not cause duplicate destroyAllWindows calls")

    def test_K3_stop_does_not_call_destroyAllWindows_directly_in_stop(self):
        import mpu.main as main_module
        system = _make_system_dev(dev_mode=False)
        destroy_calls = []

        with patch.object(main_module.cv2, "destroyAllWindows",
                          side_effect=lambda: destroy_calls.append(1)):
            system.camera.stop_capture = MagicMock()
            system.stop()

        self.assertEqual(destroy_calls, [],
                         "HelmetDetectionSystem.stop() must not call cv2.destroyAllWindows() directly")

    def test_K4_mocked_camera_stop_capture_called_once_by_system_stop(self):
        system = _make_system_dev(dev_mode=False)
        system.stop()
        system.camera.stop_capture.assert_called_once()


if __name__ == "__main__":
    unittest.main()
