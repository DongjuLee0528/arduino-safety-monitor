"""
Packaging tests for the App Lab deployment package generator.

All 14 required coverage areas:
  1.  generator creates App Lab package
  2.  generated python/mpu exists
  3.  generated mpu imports resolve when python/ is on sys.path
  4.  generated sketch/sketch.ino matches authoritative arduino/arduino.ino
  5.  generated headers match authoritative Arduino source exactly
  6.  stale generated file is removed on regeneration
  7.  authoritative mpu/ is never modified
  8.  authoritative arduino/ is never modified
  9.  app.yaml content is correct
  10. sketch.yaml content is correct
  11. generated Python main.py imports arduino.app_utils.App
  12. generated Python main.py does NOT duplicate HelmetDetectionSystem implementation
  13. generated package does not contain __pycache__ or *.pyc
  14. running generator twice produces identical generated contents

Git policy tests (10 additional):
  G1.  generated package is ignored by Git
  G2.  generated model assets are ignored by Git
  G3.  generator script is NOT ignored by Git
  G4.  packaging tests are NOT ignored by Git
  G5.  mpu/ is NOT ignored by the App Lab rule
  G6.  arduino/ is NOT ignored by the App Lab rule
  G7.  running generator twice remains deterministic (also covers git-ignore path)
  G8.  stale generated files are still removed
  G9.  generated package absence before generation is acceptable
  G10. generated package directory rule does not bleed into app_lab/ root

Dependency packaging tests (18 additional):
  D01. generator creates python/requirements.txt
  D02. requirements file is deterministic across two runs
  D03. requirements contains onnxruntime
  D04. requirements contains opencv-python-headless
  D05. requirements contains pyserial
  D06. requirements contains requests
  D07. requirements contains Pillow
  D08. requirements contains numpy
  D09. requirements does NOT contain opencv-python (bare, without -headless)
  D10. requirements does NOT contain torch
  D11. requirements does NOT contain torchvision
  D12. requirements does NOT contain ultralytics
  D13. generated requirements exactly matches tracked app_lab/requirements_app_lab.txt
  D14. generated requirements.txt is Git-ignored
  D15. tracked app_lab/requirements_app_lab.txt is NOT Git-ignored
  D16. generator fails fast when requirements_app_lab.txt is missing
  D17. generated models are still present after adding requirements
  D18. authoritative mpu/ and arduino/ are still unchanged after generation
"""

import filecmp
import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GENERATOR = REPO_ROOT / "app_lab" / "generate_app_lab.py"
AUTH_MPU = REPO_ROOT / "mpu"
AUTH_ARDUINO = REPO_ROOT / "arduino"
AUTH_INO = AUTH_ARDUINO / "arduino.ino"
SRC_REQUIREMENTS = REPO_ROOT / "app_lab" / "requirements_app_lab.txt"
SRC_UI_ASSETS = REPO_ROOT / "app_lab" / "ui" / "assets"

APP_DIR = REPO_ROOT / "app_lab" / "Arduino Safety Monitor"
PYTHON_DIR = APP_DIR / "python"
OUT_ASSETS_DIR = PYTHON_DIR / "assets"
OUT_INDEX_HTML = OUT_ASSETS_DIR / "index.html"
MPU_PACKAGE_DIR = PYTHON_DIR / "mpu"
SKETCH_DIR = APP_DIR / "sketch"
OUT_REQUIREMENTS = PYTHON_DIR / "requirements.txt"
ADAPTER = PYTHON_DIR / "main.py"
APP_YAML = APP_DIR / "app.yaml"
SKETCH_YAML = SKETCH_DIR / "sketch.yaml"
SKETCH_INO = SKETCH_DIR / "sketch.ino"
MARKER = APP_DIR / "GENERATED_FROM_SOURCE.md"

ARDUINO_HEADERS = [p.name for p in sorted(AUTH_ARDUINO.iterdir())
                   if p.suffix in {".h", ".hpp", ".cpp"}]


def _run_generator() -> None:
    spec = importlib.util.spec_from_file_location("generate_app_lab", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.generate()


def _dir_sha256(directory: Path) -> dict[str, str]:
    result = {}
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(directory))
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            result[rel] = digest
    return result


class TestGeneratorRuns(unittest.TestCase):
    def test_01_generator_creates_app_lab_package(self):
        _run_generator()
        self.assertTrue(APP_DIR.is_dir(), "App Lab directory was not created")

    def test_02_generated_python_mpu_exists(self):
        _run_generator()
        self.assertTrue(MPU_PACKAGE_DIR.is_dir(), "python/mpu/ was not created")
        self.assertTrue((MPU_PACKAGE_DIR / "__init__.py").is_file(),
                        "python/mpu/__init__.py missing")

    def test_03_generated_mpu_imports_resolve(self):
        _run_generator()
        python_dir_str = str(PYTHON_DIR)
        saved_path = sys.path[:]
        try:
            if python_dir_str not in sys.path:
                sys.path.insert(0, python_dir_str)
            for mod_name in list(sys.modules.keys()):
                if mod_name == "mpu" or mod_name.startswith("mpu."):
                    del sys.modules[mod_name]
            import mpu as _mpu_pkg  # noqa: F401
            import mpu.config as _cfg  # noqa: F401
            import mpu.bridge_rpc as _brpc  # noqa: F401
            import mpu.dashboard_state as _ds  # noqa: F401
        finally:
            sys.path[:] = saved_path
            for mod_name in list(sys.modules.keys()):
                if mod_name == "mpu" or mod_name.startswith("mpu."):
                    del sys.modules[mod_name]

    def test_04_sketch_ino_matches_authoritative(self):
        _run_generator()
        self.assertTrue(SKETCH_INO.is_file(), "sketch/sketch.ino missing")
        self.assertEqual(
            AUTH_INO.read_bytes(), SKETCH_INO.read_bytes(),
            "sketch/sketch.ino content differs from arduino/arduino.ino"
        )

    def test_05_all_headers_match_authoritative(self):
        _run_generator()
        self.assertTrue(len(ARDUINO_HEADERS) > 0, "No headers found in arduino/")
        for name in ARDUINO_HEADERS:
            auth_file = AUTH_ARDUINO / name
            gen_file = SKETCH_DIR / name
            self.assertTrue(gen_file.is_file(), f"sketch/{name} missing")
            self.assertEqual(
                auth_file.read_bytes(), gen_file.read_bytes(),
                f"sketch/{name} differs from arduino/{name}"
            )

    def test_06_stale_generated_file_removed_on_regeneration(self):
        _run_generator()
        stale = MPU_PACKAGE_DIR / "stale_canary_file_do_not_create_manually.py"
        stale.write_text("# stale\n", encoding="utf-8")
        self.assertTrue(stale.is_file(), "Failed to create stale canary file")
        _run_generator()
        self.assertFalse(stale.exists(),
                         "Stale generated file survived regeneration")

    def test_07_authoritative_mpu_not_modified(self):
        before = _dir_sha256(AUTH_MPU)
        _run_generator()
        after = _dir_sha256(AUTH_MPU)
        self.assertEqual(before, after,
                         "Generator modified files in authoritative mpu/")

    def test_08_authoritative_arduino_not_modified(self):
        before = _dir_sha256(AUTH_ARDUINO)
        _run_generator()
        after = _dir_sha256(AUTH_ARDUINO)
        self.assertEqual(before, after,
                         "Generator modified files in authoritative arduino/")

    def test_09_app_yaml_content_correct(self):
        _run_generator()
        self.assertTrue(APP_YAML.is_file(), "app.yaml missing")
        content = APP_YAML.read_text(encoding="utf-8")
        self.assertIn("name: Arduino Safety Monitor", content)
        self.assertIn("ports:", content)
        self.assertIn("bricks:", content)
        self.assertIn("arduino:web_ui", content,
                      "app.yaml must declare arduino:web_ui brick")

    def test_10_sketch_yaml_content_correct(self):
        _run_generator()
        self.assertTrue(SKETCH_YAML.is_file(), "sketch/sketch.yaml missing")
        content = SKETCH_YAML.read_text(encoding="utf-8")
        self.assertIn("arduino:zephyr", content)
        self.assertIn("default_profile: default", content)

    def test_11_adapter_imports_app(self):
        _run_generator()
        self.assertTrue(ADAPTER.is_file(), "python/main.py missing")
        content = ADAPTER.read_text(encoding="utf-8")
        self.assertIn("from arduino.app_utils import App", content)

    def test_12_adapter_does_not_duplicate_helmet_detection_system(self):
        _run_generator()
        content = ADAPTER.read_text(encoding="utf-8")
        self.assertNotIn("class HelmetDetectionSystem", content,
                         "Adapter must not redefine HelmetDetectionSystem")
        self.assertNotIn("def start(self)", content,
                         "Adapter must not duplicate start() implementation")
        self.assertNotIn("def process_frame(self", content,
                         "Adapter must not duplicate process_frame()")
        self.assertIn("from mpu.main import HelmetDetectionSystem", content,
                      "Adapter must import HelmetDetectionSystem from mpu.main")

    def test_13_no_pycache_or_pyc_in_generated_package(self):
        _run_generator()
        for f in MPU_PACKAGE_DIR.rglob("*"):
            if f.is_dir():
                self.assertNotEqual(f.name, "__pycache__",
                                    f"__pycache__ found in generated package: {f}")
            elif f.is_file():
                self.assertNotIn(f.suffix, {".pyc", ".pyo"},
                                 f"Compiled Python file found in generated package: {f}")

    def test_14_idempotent_two_runs_identical(self):
        _run_generator()
        first_run = _dir_sha256(APP_DIR)
        _run_generator()
        second_run = _dir_sha256(APP_DIR)
        self.assertEqual(first_run, second_run,
                         "Two consecutive generator runs produced different output")


def _git_is_ignored(path: str) -> bool:
    """Return True if git check-ignore reports the path as ignored."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    return result.returncode == 0


class TestGitIgnorePolicy(unittest.TestCase):
    def test_G01_generated_package_dir_is_ignored(self):
        self.assertTrue(
            _git_is_ignored("app_lab/Arduino Safety Monitor/app.yaml"),
            "app_lab/Arduino Safety Monitor/app.yaml must be git-ignored"
        )

    def test_G02_generated_model_assets_are_ignored(self):
        self.assertTrue(
            _git_is_ignored(
                "app_lab/Arduino Safety Monitor/python/mpu/ai/models/best_model.onnx"
            ),
            "Generated model assets must be git-ignored"
        )
        self.assertTrue(
            _git_is_ignored(
                "app_lab/Arduino Safety Monitor/python/mpu/ai/models/best_model.onnx.data"
            ),
            "Generated model .data file must be git-ignored"
        )

    def test_G03_generator_script_is_not_ignored(self):
        self.assertFalse(
            _git_is_ignored("app_lab/generate_app_lab.py"),
            "app_lab/generate_app_lab.py must NOT be git-ignored"
        )

    def test_G04_packaging_tests_are_not_ignored(self):
        self.assertFalse(
            _git_is_ignored("tests/test_app_lab_packaging.py"),
            "tests/test_app_lab_packaging.py must NOT be git-ignored"
        )

    def test_G05_mpu_not_ignored_by_app_lab_rule(self):
        self.assertFalse(
            _git_is_ignored("mpu/main.py"),
            "mpu/main.py must NOT be git-ignored"
        )
        self.assertFalse(
            _git_is_ignored("mpu/bridge_rpc.py"),
            "mpu/bridge_rpc.py must NOT be git-ignored"
        )

    def test_G06_arduino_not_ignored_by_app_lab_rule(self):
        self.assertFalse(
            _git_is_ignored("arduino/arduino.ino"),
            "arduino/arduino.ino must NOT be git-ignored"
        )
        self.assertFalse(
            _git_is_ignored("arduino/comm.h"),
            "arduino/comm.h must NOT be git-ignored"
        )

    def test_G07_generator_deterministic_under_ignore_policy(self):
        _run_generator()
        first = _dir_sha256(APP_DIR)
        _run_generator()
        second = _dir_sha256(APP_DIR)
        self.assertEqual(first, second,
                         "Generator output must be identical across two runs")

    def test_G08_stale_files_removed_after_regeneration(self):
        _run_generator()
        stale = SKETCH_DIR / "stale_git_policy_canary.h"
        stale.write_text("// stale\n", encoding="utf-8")
        self.assertTrue(stale.is_file())
        _run_generator()
        self.assertFalse(stale.exists(),
                         "Stale file in sketch/ must be removed on regeneration")

    def test_G09_generator_works_when_output_absent(self):
        if APP_DIR.exists():
            shutil.rmtree(APP_DIR)
        self.assertFalse(APP_DIR.exists(), "Setup: APP_DIR should be absent")
        _run_generator()
        self.assertTrue(APP_DIR.is_dir(),
                        "Generator must create package even when output was absent")
        self.assertTrue((APP_DIR / "app.yaml").is_file())

    def test_G10_ignore_rule_does_not_affect_app_lab_root_files(self):
        self.assertFalse(
            _git_is_ignored("app_lab/generate_app_lab.py"),
            "app_lab/ root files must NOT be covered by the generated-dir ignore rule"
        )


class TestDependencyPackaging(unittest.TestCase):
    def test_D01_generator_creates_requirements_txt(self):
        _run_generator()
        self.assertTrue(OUT_REQUIREMENTS.is_file(),
                        "python/requirements.txt was not created")

    def test_D02_requirements_deterministic(self):
        _run_generator()
        first = OUT_REQUIREMENTS.read_bytes()
        _run_generator()
        second = OUT_REQUIREMENTS.read_bytes()
        self.assertEqual(first, second,
                         "requirements.txt differs between two generator runs")

    def _req_lines(self) -> set[str]:
        _run_generator()
        lines = OUT_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        return {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}

    def test_D03_requires_onnxruntime(self):
        pkgs = self._req_lines()
        self.assertTrue(any("onnxruntime" in p for p in pkgs),
                        f"onnxruntime missing from requirements: {pkgs}")

    def test_D04_requires_opencv_headless(self):
        pkgs = self._req_lines()
        self.assertTrue(any("opencv-python-headless" in p for p in pkgs),
                        f"opencv-python-headless missing from requirements: {pkgs}")

    def test_D05_requires_pyserial(self):
        pkgs = self._req_lines()
        self.assertTrue(any("pyserial" in p for p in pkgs),
                        f"pyserial missing from requirements: {pkgs}")

    def test_D06_requires_requests(self):
        pkgs = self._req_lines()
        self.assertTrue(any("requests" in p for p in pkgs),
                        f"requests missing from requirements: {pkgs}")

    def test_D07_requires_pillow(self):
        pkgs = self._req_lines()
        self.assertTrue(any("Pillow" in p or "pillow" in p for p in pkgs),
                        f"Pillow missing from requirements: {pkgs}")

    def test_D08_requires_numpy(self):
        pkgs = self._req_lines()
        self.assertTrue(any("numpy" in p for p in pkgs),
                        f"numpy missing from requirements: {pkgs}")

    def test_D09_no_bare_opencv_python(self):
        pkgs = self._req_lines()
        bare = {p for p in pkgs
                if "opencv-python" in p and "headless" not in p}
        self.assertEqual(bare, set(),
                         f"Bare opencv-python (non-headless) must not appear: {bare}")

    def test_D10_no_torch(self):
        pkgs = self._req_lines()
        self.assertFalse(any(p == "torch" or p.startswith("torch>") or
                             p.startswith("torch<") or p.startswith("torch=")
                             for p in pkgs),
                         f"torch must not appear in App Lab requirements: {pkgs}")

    def test_D11_no_torchvision(self):
        pkgs = self._req_lines()
        self.assertFalse(any("torchvision" in p for p in pkgs),
                         f"torchvision must not appear in App Lab requirements: {pkgs}")

    def test_D12_no_ultralytics(self):
        pkgs = self._req_lines()
        self.assertFalse(any("ultralytics" in p for p in pkgs),
                         f"ultralytics must not appear in App Lab requirements: {pkgs}")

    def test_D13_generated_matches_tracked_source(self):
        _run_generator()
        generated = OUT_REQUIREMENTS.read_text(encoding="utf-8")
        tracked = SRC_REQUIREMENTS.read_text(encoding="utf-8")
        self.assertEqual(generated, tracked,
                         "Generated requirements.txt does not match app_lab/requirements_app_lab.txt")

    def test_D14_generated_requirements_is_git_ignored(self):
        self.assertTrue(
            _git_is_ignored("app_lab/Arduino Safety Monitor/python/requirements.txt"),
            "Generated python/requirements.txt must be git-ignored"
        )

    def test_D15_tracked_requirements_source_not_ignored(self):
        self.assertFalse(
            _git_is_ignored("app_lab/requirements_app_lab.txt"),
            "app_lab/requirements_app_lab.txt must NOT be git-ignored"
        )

    def test_D16_generator_fails_fast_without_requirements_source(self):
        import importlib.util as _ilu
        import sys as _sys
        import io

        original = SRC_REQUIREMENTS.read_text(encoding="utf-8")
        SRC_REQUIREMENTS.rename(SRC_REQUIREMENTS.with_suffix(".txt.bak"))
        bak = SRC_REQUIREMENTS.with_suffix(".txt.bak")
        try:
            spec = _ilu.spec_from_file_location("generate_app_lab_d16", GENERATOR)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with self.assertRaises(SystemExit) as ctx:
                mod.generate()
            self.assertNotEqual(ctx.exception.code, 0,
                                "Generator must exit non-zero when requirements source is missing")
        finally:
            bak.rename(SRC_REQUIREMENTS)

    def test_D17_models_present_after_requirements_added(self):
        _run_generator()
        models_dir = MPU_PACKAGE_DIR / "ai" / "models"
        self.assertTrue(models_dir.is_dir(), "ai/models/ directory missing")
        self.assertTrue(any(models_dir.iterdir()),
                        "ai/models/ directory is empty")

    def test_D18_sources_unchanged_after_requirements_generation(self):
        before_mpu = _dir_sha256(AUTH_MPU)
        before_arduino = _dir_sha256(AUTH_ARDUINO)
        _run_generator()
        self.assertEqual(before_mpu, _dir_sha256(AUTH_MPU),
                         "mpu/ was modified during generation")
        self.assertEqual(before_arduino, _dir_sha256(AUTH_ARDUINO),
                         "arduino/ was modified during generation")


class TestUIPackaging(unittest.TestCase):
    """
    UI packaging tests (U01–U11).

    U01. generator copies assets/index.html into generated package.
    U02. generated assets/index.html is non-empty.
    U03. generated assets/index.html contains Dashboard heading.
    U04. generated assets/index.html contains /api/state endpoint reference.
    U05. generated assets/index.html does NOT contain fake detection data.
    U06. generated app.yaml declares arduino:web_ui brick.
    U07. adapter imports WebUI from arduino.app_bricks.web_ui.
    U08. adapter exposes /api/state endpoint.
    U09. adapter control endpoints check hardware availability before acting.
    U10. generated assets directory is git-ignored.
    U11. authoritative ui/assets/ source is NOT git-ignored.
    """

    def test_U01_generator_copies_assets_index_html(self):
        _run_generator()
        self.assertTrue(
            OUT_INDEX_HTML.is_file(),
            f"generated assets/index.html not found at {OUT_INDEX_HTML}"
        )

    def test_U02_generated_index_html_is_non_empty(self):
        _run_generator()
        size = OUT_INDEX_HTML.stat().st_size
        self.assertGreater(size, 0, "generated index.html is empty")

    def test_U03_generated_index_html_contains_dashboard_heading(self):
        _run_generator()
        content = OUT_INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("Arduino Safety Monitor", content,
                      "index.html must contain the application title")

    def test_U04_generated_index_html_references_api_state(self):
        _run_generator()
        content = OUT_INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("/api/state", content,
                      "index.html must fetch /api/state for live data")

    def test_U05_generated_index_html_no_fake_detection_data(self):
        _run_generator()
        content = OUT_INDEX_HTML.read_text(encoding="utf-8")
        fake_indicators = [
            "confidence: 0.9",
            "confidence: 0.8",
            "confidence: 0.7",
            "worker_1",
            "fake_detection",
            "mock_frame",
        ]
        for indicator in fake_indicators:
            self.assertNotIn(
                indicator, content,
                f"index.html must not contain fake detection data: '{indicator}'"
            )

    def test_U06_app_yaml_declares_web_ui_brick(self):
        _run_generator()
        content = APP_YAML.read_text(encoding="utf-8")
        self.assertIn("arduino:web_ui", content,
                      "app.yaml must declare arduino:web_ui brick for dashboard UI")

    def test_U07_adapter_imports_web_ui_brick(self):
        _run_generator()
        content = ADAPTER.read_text(encoding="utf-8")
        self.assertIn("from arduino.app_bricks.web_ui import WebUI", content,
                      "Adapter must import WebUI from arduino.app_bricks.web_ui")

    def test_U08_adapter_exposes_api_state_endpoint(self):
        _run_generator()
        content = ADAPTER.read_text(encoding="utf-8")
        self.assertIn("/api/state", content,
                      "Adapter must register /api/state API endpoint")
        self.assertIn("expose_api", content,
                      "Adapter must call expose_api to register endpoints")

    def test_U09_control_endpoints_check_hardware_before_acting(self):
        _run_generator()
        content = ADAPTER.read_text(encoding="utf-8")
        self.assertIn("Hardware not connected", content,
                      "Control endpoints must return error when hardware not connected")
        self.assertIn("_connected", content,
                      "Control endpoints must check _connected before sending commands")

    def test_U10_generated_assets_dir_is_git_ignored(self):
        self.assertTrue(
            _git_is_ignored(
                "app_lab/Arduino Safety Monitor/python/assets/index.html"
            ),
            "Generated python/assets/index.html must be git-ignored"
        )

    def test_U11_authoritative_ui_assets_not_ignored(self):
        self.assertFalse(
            _git_is_ignored("app_lab/ui/assets/index.html"),
            "app_lab/ui/assets/index.html must NOT be git-ignored"
        )


class TestUIState(unittest.TestCase):
    """
    Tests for mpu/ui_state.py view-model bridge (S01–S08).

    S01. build_ui_payload returns expected top-level keys.
    S02. dev_mode=True is reflected in payload.
    S03. camera_available=False shows unavailable in info.
    S04. camera_available=True shows available in info.
    S05. disconnected BridgeRPC shows offline in state (not faked as connected).
    S06. missing detection does not generate fake confidence.
    S07. server URL with credentials has them redacted.
    S08. payload does not expose full environment dump or secrets.
    """

    def _empty_snap(self):
        from mpu.dashboard_state import DashboardState
        return DashboardState().snapshot()

    def test_S01_payload_has_expected_keys(self):
        from mpu.ui_state import build_ui_payload
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=False,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000/api/alert",
        )
        for key in ("state", "dev_mode", "warning_active", "info"):
            self.assertIn(key, payload, f"Missing key: {key}")

    def test_S02_dev_mode_true_reflected_in_payload(self):
        from mpu.ui_state import build_ui_payload
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=True,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
        )
        self.assertTrue(payload["dev_mode"])

    def test_S03_camera_unavailable_shown_in_info(self):
        from mpu.ui_state import build_ui_payload
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=True,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
        )
        self.assertFalse(payload["info"]["camera_available"])

    def test_S04_camera_available_shown_in_info(self):
        from mpu.ui_state import build_ui_payload
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=False,
            camera_available=True,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
        )
        self.assertTrue(payload["info"]["camera_available"])

    def test_S05_disconnected_state_is_offline_not_faked(self):
        from mpu.ui_state import build_ui_payload
        snap = self._empty_snap()
        payload = build_ui_payload(
            snap,
            dev_mode=False,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
        )
        conn_status = payload["state"]["connection"]["status"]
        self.assertEqual(conn_status, "offline",
                         "Disconnected system must show 'offline', not fake 'online'")

    def test_S06_no_detection_no_fake_confidence(self):
        from mpu.ui_state import build_ui_payload
        snap = self._empty_snap()
        payload = build_ui_payload(
            snap,
            dev_mode=False,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
        )
        det = payload["state"]["detection"]
        self.assertFalse(det["worker_present"],
                         "No-detection state must not show worker_present=True")
        self.assertEqual(det["helmet_result"], "unknown",
                         "No-detection must not produce a definitive helmet result")

    def test_S07_server_url_with_credentials_redacted(self):
        from mpu.ui_state import build_ui_payload
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=False,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://admin:secret@localhost:3000/api/alert",
        )
        url_in_payload = payload["info"]["server_url"]
        self.assertNotIn("secret", url_in_payload,
                         "Credentials must not appear in server_url payload")
        self.assertNotIn("admin", url_in_payload,
                         "Username must not appear in server_url payload")

    def test_S08_payload_does_not_expose_env_dump(self):
        from mpu.ui_state import build_ui_payload
        import json
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=True,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
        )
        payload_str = json.dumps(payload)
        env_indicators = ["PATH", "HOME", "USER", "SHELL", "PYTHONPATH"]
        for key in env_indicators:
            self.assertNotIn(
                f'"{key}"', payload_str,
                f"Payload must not expose environment variable: {key}"
            )


class TestCodeReviewFixes(unittest.TestCase):
    """
    Code review fix tests (CR01–CR12).

    F-001 – Motion Lease / Control Tick truthfulness:
      CR01. index.html does NOT set Motion Lease to ACTIVE when conn === online.
      CR02. index.html does NOT set Control Tick to RUNNING when conn === online.
      CR03. index.html uses UNKNOWN for Motion Lease in the online branch.
      CR04. index.html uses UNKNOWN for Control Tick in the online branch.

    F-002 – D3 LED Warning State:
      CR05. build_ui_payload warning_active=None passes null through (not False).
      CR06. build_ui_payload warning_active=True passes True through.
      CR07. build_ui_payload warning_active=False passes False through.
      CR08. index.html does NOT hard-code 'OFF' as default D3 LED pill text.
      CR09. index.html renders UNKNOWN when warning_active is null/none.

    F-003 – Buzzer Test API:
      CR10. generated adapter does NOT call buzzer_control(True).
      CR11. generated adapter returns unsupported error for test_buzzer.

    F-004 – Generator Stale Artifact Cleanup:
      CR12. stale file in python/ root is removed on regeneration.
      CR13. path guard rejects APP_DIR == REPO_ROOT.
      CR14. path guard rejects APP_DIR == APP_LAB_DIR.
      CR15. two runs remain deterministic after full-dir clear.
    """

    def _empty_snap(self):
        from mpu.dashboard_state import DashboardState
        return DashboardState().snapshot()

    def test_CR01_motion_lease_not_ACTIVE_when_online(self):
        content = SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")
        import re
        online_block_match = re.search(
            r'if \(conn === "online"\)(.*?)(?=\} else if \(devMode\))',
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(online_block_match, "Could not find the conn===online block")
        online_block = online_block_match.group(1)
        self.assertNotIn(
            '"ACTIVE"',
            online_block,
            "Motion Lease must NOT be set to ACTIVE based on connection status alone"
        )

    def test_CR02_control_tick_not_RUNNING_when_online(self):
        content = SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")
        import re
        online_block_match = re.search(
            r'if \(conn === "online"\)(.*?)(?=\} else if \(devMode\))',
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(online_block_match, "Could not find the conn===online block")
        online_block = online_block_match.group(1)
        self.assertNotIn(
            '"RUNNING"',
            online_block,
            "Control Tick must NOT be set to RUNNING based on connection status alone"
        )

    def test_CR03_motion_lease_shows_UNKNOWN_when_online(self):
        content = SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")
        import re
        pills_blocks = list(re.finditer(
            r'if \(conn === "online"\)(.*?)(?=\} else if \(devMode\))',
            content,
            re.DOTALL,
        ))
        pills_block = next(
            (m.group(1) for m in pills_blocks if "pill-lease" in m.group(1)),
            None,
        )
        self.assertIsNotNone(pills_block, "Could not find the pills conn===online block containing pill-lease")
        lease_line_match = re.search(r'pill-lease.*?"([A-Z /]+)"', pills_block)
        self.assertIsNotNone(lease_line_match, "Could not find pill-lease assignment")
        label = lease_line_match.group(1)
        self.assertIn(label, {"UNKNOWN", "N/A", "INACTIVE"},
                      f"Motion Lease online label must be UNKNOWN/N/A/INACTIVE, got: {label}")

    def test_CR04_control_tick_shows_UNKNOWN_when_online(self):
        content = SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")
        import re
        pills_blocks = list(re.finditer(
            r'if \(conn === "online"\)(.*?)(?=\} else if \(devMode\))',
            content,
            re.DOTALL,
        ))
        pills_block = next(
            (m.group(1) for m in pills_blocks if "pill-tick" in m.group(1)),
            None,
        )
        self.assertIsNotNone(pills_block, "Could not find the pills conn===online block containing pill-tick")
        tick_line_match = re.search(r'pill-tick.*?"([A-Z /]+)"', pills_block)
        self.assertIsNotNone(tick_line_match, "Could not find pill-tick assignment")
        label = tick_line_match.group(1)
        self.assertIn(label, {"UNKNOWN", "N/A", "INACTIVE"},
                      f"Control Tick online label must be UNKNOWN/N/A/INACTIVE, got: {label}")

    def test_CR05_warning_active_none_passes_null(self):
        from mpu.ui_state import build_ui_payload
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=False,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
            warning_active=None,
        )
        self.assertIsNone(payload["warning_active"],
                          "warning_active=None must remain None in payload (not coerced to False)")

    def test_CR06_warning_active_true_passes_through(self):
        from mpu.ui_state import build_ui_payload
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=False,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
            warning_active=True,
        )
        self.assertIs(payload["warning_active"], True,
                      "warning_active=True must be True in payload")

    def test_CR07_warning_active_false_passes_through(self):
        from mpu.ui_state import build_ui_payload
        payload = build_ui_payload(
            self._empty_snap(),
            dev_mode=False,
            camera_available=False,
            serial_port="/dev/ttyUSB0",
            server_url="http://localhost:3000",
            warning_active=False,
        )
        self.assertIs(payload["warning_active"], False,
                      "warning_active=False must be False in payload")

    def test_CR08_index_html_d3_led_default_not_OFF(self):
        content = SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")
        import re
        pill_led_match = re.search(
            r'id="pill-led"[^>]*>(.*?)</span>',
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(pill_led_match, "Could not find pill-led span in HTML")
        pill_text = pill_led_match.group(0)
        self.assertNotIn(
            ">OFF<",
            pill_text,
            "D3 LED default pill text must not be OFF — should be UNKNOWN when state is unavailable"
        )

    def test_CR09_index_html_warning_null_renders_UNKNOWN(self):
        content = SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")
        self.assertIn(
            "UNKNOWN",
            content,
            "index.html must render UNKNOWN for D3 LED when warning_active is null"
        )
        self.assertIn(
            "=== null",
            content,
            "index.html must check for null (=== null) for warning_active"
        )

    def test_CR10_adapter_does_not_call_buzzer_control_bool(self):
        _run_generator()
        content = ADAPTER.read_text(encoding="utf-8")
        self.assertNotIn(
            "buzzer_control(True)",
            content,
            "Adapter must not pass a boolean to buzzer_control"
        )
        self.assertNotIn(
            "buzzer_control(False)",
            content,
            "Adapter must not pass a boolean to buzzer_control"
        )

    def test_CR11_adapter_test_buzzer_returns_unsupported(self):
        _run_generator()
        content = ADAPTER.read_text(encoding="utf-8")
        self.assertIn(
            "unsupported",
            content,
            "test_buzzer endpoint must return unsupported error (no safe off lifecycle)"
        )

    def test_CR12_stale_python_root_file_removed_on_regeneration(self):
        _run_generator()
        stale = PYTHON_DIR / "stale_root_canary.py"
        stale.write_text("# stale\n", encoding="utf-8")
        self.assertTrue(stale.is_file(), "Failed to create stale file in python/ root")
        _run_generator()
        self.assertFalse(
            stale.exists(),
            "Stale file in python/ root must be removed on regeneration (full-dir clear)"
        )

    def test_CR13_path_guard_rejects_repo_root(self):
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("generate_app_lab_cr13", GENERATOR)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        original = mod.APP_DIR
        mod.APP_DIR = REPO_ROOT
        try:
            with self.assertRaises(SystemExit) as ctx:
                mod._clear_app_dir()
            self.assertNotEqual(ctx.exception.code, 0,
                                "Path guard must exit non-zero for REPO_ROOT target")
        finally:
            mod.APP_DIR = original

    def test_CR14_path_guard_rejects_app_lab_dir(self):
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("generate_app_lab_cr14", GENERATOR)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        original = mod.APP_DIR
        mod.APP_DIR = REPO_ROOT / "app_lab"
        try:
            with self.assertRaises(SystemExit) as ctx:
                mod._clear_app_dir()
            self.assertNotEqual(ctx.exception.code, 0,
                                "Path guard must exit non-zero for APP_LAB_DIR target")
        finally:
            mod.APP_DIR = original

    def test_CR15_deterministic_after_full_dir_clear(self):
        _run_generator()
        first = _dir_sha256(APP_DIR)
        _run_generator()
        second = _dir_sha256(APP_DIR)
        self.assertEqual(first, second,
                         "Two consecutive runs must produce identical output after full-dir clear")


if __name__ == "__main__":
    unittest.main()
