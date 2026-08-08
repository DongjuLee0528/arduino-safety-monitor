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


if __name__ == "__main__":
    unittest.main()
