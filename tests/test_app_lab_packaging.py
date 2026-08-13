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
OUT_ASSETS_DIR = APP_DIR / "assets"
OLD_OUT_ASSETS_DIR = PYTHON_DIR / "assets"
OUT_INDEX_HTML = OUT_ASSETS_DIR / "index.html"
OLD_OUT_INDEX_HTML = OLD_OUT_ASSETS_DIR / "index.html"
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
    UI packaging tests (U01–U13).

    U01. generator copies assets/index.html into generated package.
    U02. generated assets/index.html is non-empty.
    U03. generated assets/index.html contains Dashboard heading.
    U04. generated assets/index.html contains /api/state endpoint reference.
    U05. generated assets/index.html does NOT contain fake detection data.
    U06. generated app.yaml declares arduino:web_ui brick.
    U07. adapter imports WebUI from arduino.app_bricks.web_ui.
    U08. adapter exposes /api/state endpoint.
    U09. adapter control endpoints check hardware availability before acting.
    U10. legacy generated python/assets/index.html does not exist.
    U11. generated assets match authoritative UI source.
    U12. generated assets directory is git-ignored.
    U13. authoritative ui/assets/ source is NOT git-ignored.
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

    def test_U10_legacy_python_assets_index_html_absent(self):
        _run_generator()
        self.assertFalse(
            OLD_OUT_INDEX_HTML.exists(),
            "legacy python/assets/index.html must not exist in generated App Lab package",
        )

    def test_U11_generated_assets_match_authoritative_source(self):
        _run_generator()
        self.assertEqual(
            SRC_UI_ASSETS.joinpath("index.html").read_bytes(),
            OUT_INDEX_HTML.read_bytes(),
            "generated assets/index.html must match app_lab/ui/assets/index.html",
        )

    def test_U12_generated_assets_dir_is_git_ignored(self):
        self.assertTrue(
            _git_is_ignored(
                "app_lab/Arduino Safety Monitor/assets/index.html"
            ),
            "Generated assets/index.html must be git-ignored"
        )

    def test_U13_authoritative_ui_assets_not_ignored(self):
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

    def test_CR12b_stale_legacy_python_assets_removed_on_regeneration(self):
        _run_generator()
        stale = OLD_OUT_ASSETS_DIR / "index.html"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("stale legacy asset\n", encoding="utf-8")
        self.assertTrue(stale.is_file(), "Failed to create stale legacy python/assets file")
        _run_generator()
        self.assertFalse(
            OLD_OUT_ASSETS_DIR.exists(),
            "Legacy python/assets directory must be removed by full generated-dir cleanup"
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


class TestAppLabDevModeActivation(unittest.TestCase):
    """
    Real App Lab runtime boundary tests (ALD01-ALD07).

    ALD01. generated adapter sets APP_LAB_DEV_MODE before importing mpu.main.
    ALD02. unset environment defaults generated adapter to dev mode enabled.
    ALD03. explicit APP_LAB_DEV_MODE=false is respected.
    ALD04. production mpu/config.py default remains strict when env is unset.
    ALD05. generated adapter dev mode allows camera-unavailable construction.
    ALD06. generated /api/state reports the same dev-mode value the runtime read.
    ALD07. generated requirements remain derived from requirements_app_lab.txt.
    """

    _UNSET = object()

    def _clear_mpu_modules(self):
        for mod_name in list(sys.modules.keys()):
            if mod_name == "mpu" or mod_name.startswith("mpu."):
                del sys.modules[mod_name]

    def _load_generated_adapter(self, env_value=_UNSET):
        import runpy

        _run_generator()
        saved_path = sys.path[:]
        saved_env = os.environ.get("APP_LAB_DEV_MODE", self._UNSET)
        saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "arduino",
                "arduino.app_utils",
                "arduino.app_bricks",
                "arduino.app_bricks.web_ui",
            )
        }

        class FakeApp:
            user_loop = None

            @staticmethod
            def run(user_loop):
                FakeApp.user_loop = user_loop

        class FakeWebUI:
            def __init__(self):
                self.routes = []

            def expose_api(self, method, path, handler):
                self.routes.append((method, path, handler))

        arduino_mod = types.ModuleType("arduino")
        app_utils_mod = types.ModuleType("arduino.app_utils")
        app_bricks_mod = types.ModuleType("arduino.app_bricks")
        web_ui_mod = types.ModuleType("arduino.app_bricks.web_ui")
        app_utils_mod.App = FakeApp
        web_ui_mod.WebUI = FakeWebUI

        try:
            if env_value is self._UNSET:
                os.environ.pop("APP_LAB_DEV_MODE", None)
            else:
                os.environ["APP_LAB_DEV_MODE"] = env_value

            sys.modules["arduino"] = arduino_mod
            sys.modules["arduino.app_utils"] = app_utils_mod
            sys.modules["arduino.app_bricks"] = app_bricks_mod
            sys.modules["arduino.app_bricks.web_ui"] = web_ui_mod

            sys.path.insert(0, str(PYTHON_DIR))
            self._clear_mpu_modules()
            namespace = runpy.run_path(str(ADAPTER))
            runtime_mode = namespace["APP_LAB_DEV_MODE"]
            api_state = namespace["_api_state"]
            payload = api_state()
            return runtime_mode, payload, ADAPTER.read_text(encoding="utf-8")
        finally:
            self._clear_mpu_modules()
            sys.path[:] = saved_path
            if saved_env is self._UNSET:
                os.environ.pop("APP_LAB_DEV_MODE", None)
            else:
                os.environ["APP_LAB_DEV_MODE"] = saved_env
            for name, mod in saved_modules.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def _load_adapter_with_fake_runtime(self, *, dev_mode: bool):
        import runpy

        _run_generator()
        saved_path = sys.path[:]
        saved_env = os.environ.get("APP_LAB_DEV_MODE", self._UNSET)
        module_names = (
            "arduino",
            "arduino.app_utils",
            "arduino.app_bricks",
            "arduino.app_bricks.web_ui",
            "mpu",
            "mpu.main",
            "mpu.config",
            "mpu.ui_state",
        )
        saved_modules = {name: sys.modules.get(name) for name in module_names}

        class FakeApp:
            user_loop = None

            @staticmethod
            def run(user_loop):
                FakeApp.user_loop = user_loop

        class FakeWebUI:
            def expose_api(self, method, path, handler):
                pass

        class FakeDashboard:
            def snapshot(self):
                return {"connection": {"status": "offline"}, "events": []}

        class FakeHelmetDetectionSystem:
            def __init__(self, port=None, server_url=None):
                self.running = True
                self.dashboard = FakeDashboard()
                self.camera = types.SimpleNamespace(_camera_available=False)
                self._connected = False
                self.bridge_rpc = types.SimpleNamespace()

            def start(self):
                self.running = False

            def stop(self):
                self.running = False

        def fake_build_ui_payload(snapshot, **kwargs):
            return {"state": snapshot, "dev_mode": kwargs["dev_mode"]}

        arduino_mod = types.ModuleType("arduino")
        app_utils_mod = types.ModuleType("arduino.app_utils")
        app_bricks_mod = types.ModuleType("arduino.app_bricks")
        web_ui_mod = types.ModuleType("arduino.app_bricks.web_ui")
        app_utils_mod.App = FakeApp
        web_ui_mod.WebUI = FakeWebUI

        mpu_mod = types.ModuleType("mpu")
        main_mod = types.ModuleType("mpu.main")
        config_mod = types.ModuleType("mpu.config")
        ui_state_mod = types.ModuleType("mpu.ui_state")
        main_mod.HelmetDetectionSystem = FakeHelmetDetectionSystem
        config_mod.DEFAULT_SERIAL_PORT = "/dev/null"
        config_mod.DEFAULT_SERVER_URL = "http://localhost"
        config_mod.APP_LAB_DEV_MODE = dev_mode
        config_mod.validate_runtime_models = lambda: None
        ui_state_mod.build_ui_payload = fake_build_ui_payload

        try:
            os.environ["APP_LAB_DEV_MODE"] = "true" if dev_mode else "false"
            sys.modules["arduino"] = arduino_mod
            sys.modules["arduino.app_utils"] = app_utils_mod
            sys.modules["arduino.app_bricks"] = app_bricks_mod
            sys.modules["arduino.app_bricks.web_ui"] = web_ui_mod
            sys.modules["mpu"] = mpu_mod
            sys.modules["mpu.main"] = main_mod
            sys.modules["mpu.config"] = config_mod
            sys.modules["mpu.ui_state"] = ui_state_mod
            sys.path.insert(0, str(PYTHON_DIR))
            namespace = runpy.run_path(str(ADAPTER))
            return namespace
        finally:
            sys.path[:] = saved_path
            if saved_env is self._UNSET:
                os.environ.pop("APP_LAB_DEV_MODE", None)
            else:
                os.environ["APP_LAB_DEV_MODE"] = saved_env
            for name, mod in saved_modules.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def test_ALD01_adapter_sets_env_before_mpu_imports(self):
        _run_generator()
        content = ADAPTER.read_text(encoding="utf-8")
        setdefault_pos = content.find('os.environ.setdefault("APP_LAB_DEV_MODE", "true")')
        mpu_main_pos = content.find("from mpu.main import HelmetDetectionSystem")
        mpu_config_pos = content.find("from mpu.config import")
        self.assertNotEqual(setdefault_pos, -1,
                            "generated adapter must default APP_LAB_DEV_MODE")
        self.assertNotEqual(mpu_main_pos, -1,
                            "generated adapter must import mpu.main")
        self.assertNotEqual(mpu_config_pos, -1,
                            "generated adapter must import mpu.config")
        self.assertLess(setdefault_pos, mpu_main_pos,
                        "APP_LAB_DEV_MODE must be configured before mpu.main import")
        self.assertLess(setdefault_pos, mpu_config_pos,
                        "APP_LAB_DEV_MODE must be configured before mpu.config import")

    def test_ALD02_unset_env_defaults_generated_adapter_to_dev_mode(self):
        runtime_mode, payload, _content = self._load_generated_adapter()
        self.assertTrue(runtime_mode)
        self.assertTrue(payload["dev_mode"])

    def test_ALD03_explicit_false_is_respected_by_generated_adapter(self):
        runtime_mode, payload, _content = self._load_generated_adapter("false")
        self.assertFalse(runtime_mode)
        self.assertFalse(payload["dev_mode"])

    def test_ALD04_production_config_default_remains_strict(self):
        env = os.environ.copy()
        env.pop("APP_LAB_DEV_MODE", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mpu.config as cfg; print(cfg.APP_LAB_DEV_MODE)",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_ALD05_generated_dev_mode_allows_camera_unavailable(self):
        runtime_mode, _payload, _content = self._load_generated_adapter()
        from unittest.mock import MagicMock, patch
        from mpu.camera import CameraCapture

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap
            cam = CameraCapture(camera_index=0, dev_mode=runtime_mode)
        self.assertFalse(cam._camera_available)

    def test_ALD06_generated_state_payload_matches_runtime_mode(self):
        runtime_mode, payload, _content = self._load_generated_adapter()
        self.assertIs(payload["dev_mode"], runtime_mode)

    def test_ALD07_generated_requirements_still_match_source(self):
        _run_generator()
        self.assertEqual(
            OUT_REQUIREMENTS.read_text(encoding="utf-8"),
            SRC_REQUIREMENTS.read_text(encoding="utf-8"),
        )

    def test_ALD08_dev_worker_completion_logged_once_without_warning_spam(self):
        namespace = self._load_adapter_with_fake_runtime(dev_mode=True)
        user_loop = namespace["user_loop"]
        user_loop()
        user_loop.__globals__["_worker_thread"].join(timeout=2.0)
        with self.assertLogs(level="INFO") as log_ctx:
            user_loop()
            user_loop()
            user_loop()
        messages = "\n".join(log_ctx.output)
        self.assertEqual(messages.count("dev-mode worker completed as expected"), 1)
        self.assertNotIn("WARNING", messages)
        self.assertNotIn("worker thread has exited unexpectedly", messages)
        self.assertNotIn("HelmetDetectionSystem.running is False", messages)

    def test_ALD09_strict_worker_exit_remains_observable(self):
        namespace = self._load_adapter_with_fake_runtime(dev_mode=False)
        user_loop = namespace["user_loop"]
        user_loop()
        user_loop.__globals__["_worker_thread"].join(timeout=2.0)
        with self.assertLogs(level="WARNING") as log_ctx:
            user_loop()
        messages = "\n".join(log_ctx.output)
        self.assertIn("worker thread has exited unexpectedly", messages)


class TestUIV2Structure(unittest.TestCase):
    """
    UI v2 responsive dashboard structure tests (V01–V15).

    V01.  authoritative ui/assets/index.html exists.
    V02.  generated assets/index.html exists after generation.
    V03.  legacy python/assets/index.html does NOT exist.
    V04.  viewport meta tag is present (width=device-width, initial-scale=1).
    V05.  at least three responsive @media query breakpoints are present.
    V06.  no hard-coded desktop-only width assumptions (no min-width:1200px body).
    V07.  #app-sidebar element exists for drawer/nav structure.
    V08.  hamburger button (#hamburger) exists with aria-expanded attribute.
    V09.  mobile nav overlay (#nav-overlay) exists.
    V10.  tablet/mobile inline panel classes are represented in HTML.
    V11.  summary-grid class is present for responsive detection summary.
    V12.  no fake hardware data introduced in v2.
    V13.  /api/state integration is present.
    V14.  all control endpoints are wired (stop, reset, estop, test_led, test_buzzer).
    V15.  emergency stop present for both desktop and mobile (btn-estop, btn-estop-m).
    V16.  desktop hidden inline-panel rule comes before tablet/mobile display overrides.
    """

    def _src_content(self):
        return SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")

    def _gen_content(self):
        _run_generator()
        return OUT_INDEX_HTML.read_text(encoding="utf-8")

    def test_V01_authoritative_ui_source_exists(self):
        self.assertTrue(
            SRC_UI_ASSETS.joinpath("index.html").is_file(),
            "app_lab/ui/assets/index.html must exist as authoritative UI source"
        )

    def test_V02_generated_assets_index_html_exists(self):
        _run_generator()
        self.assertTrue(
            OUT_INDEX_HTML.is_file(),
            f"generated assets/index.html not found at {OUT_INDEX_HTML}"
        )

    def test_V03_no_legacy_python_assets_path(self):
        _run_generator()
        self.assertFalse(
            OLD_OUT_INDEX_HTML.exists(),
            "legacy python/assets/index.html must not exist in generated package"
        )
        self.assertFalse(
            OLD_OUT_ASSETS_DIR.exists(),
            "legacy python/assets/ directory must not exist in generated package"
        )

    def test_V04_viewport_meta_present(self):
        content = self._src_content()
        self.assertIn(
            'name="viewport"',
            content,
            "index.html must contain a viewport meta tag for responsive behaviour"
        )
        self.assertIn(
            "width=device-width",
            content,
            "viewport meta must set width=device-width"
        )
        self.assertIn(
            "initial-scale=1",
            content,
            "viewport meta must set initial-scale=1"
        )

    def test_V05_responsive_media_queries_present(self):
        import re
        content = self._src_content()
        queries = re.findall(r'@media\s*\(', content)
        self.assertGreaterEqual(
            len(queries), 3,
            f"index.html must have at least 3 responsive @media queries, found {len(queries)}"
        )

    def test_V06_no_desktop_only_fixed_layout(self):
        content = self._src_content()
        self.assertNotIn(
            "min-width: 1200px",
            content.replace("min-width:1200px", "min-width: 1200px"),
        )

    def test_V07_sidebar_nav_element_exists(self):
        content = self._src_content()
        self.assertIn(
            'id="app-sidebar"',
            content,
            "index.html must have #app-sidebar element for sidebar/drawer navigation"
        )

    def test_V08_hamburger_button_with_aria_expanded(self):
        content = self._src_content()
        self.assertIn(
            'id="hamburger"',
            content,
            "index.html must have a #hamburger button for mobile navigation"
        )
        self.assertIn(
            'aria-expanded=',
            content,
            "hamburger button must have aria-expanded attribute"
        )
        self.assertIn(
            'aria-controls="app-sidebar"',
            content,
            "hamburger button must have aria-controls pointing to app-sidebar"
        )

    def test_V09_mobile_nav_overlay_exists(self):
        content = self._src_content()
        self.assertIn(
            'id="nav-overlay"',
            content,
            "index.html must have #nav-overlay element for mobile drawer backdrop"
        )

    def test_V10_tablet_mobile_inline_panel_classes_present(self):
        content = self._src_content()
        for cls in ("tablet-system-status", "tablet-controls", "tablet-quick-info"):
            self.assertIn(
                cls, content,
                f"index.html must use .{cls} class for tablet/mobile inline panels"
            )

    def test_V11_summary_grid_class_present(self):
        content = self._src_content()
        self.assertIn(
            "summary-grid",
            content,
            "index.html must use .summary-grid class for responsive detection summary"
        )

    def test_V12_no_fake_detection_data_in_v2(self):
        content = self._src_content()
        for indicator in [
            "confidence: 0.9", "confidence: 0.8", "worker_1",
            "fake_detection", "mock_frame",
        ]:
            self.assertNotIn(
                indicator, content,
                f"v2 index.html must not contain fake detection data: '{indicator}'"
            )

    def test_V13_api_state_integration_present(self):
        content = self._src_content()
        self.assertIn(
            "/api/state",
            content,
            "index.html must fetch /api/state for live data"
        )

    def test_V14_all_control_endpoints_wired(self):
        content = self._src_content()
        for endpoint in [
            "/api/control/stop",
            "/api/control/reset",
            "/api/control/estop",
            "/api/control/test_led",
            "/api/control/test_buzzer",
        ]:
            self.assertIn(
                endpoint, content,
                f"index.html must wire control endpoint: {endpoint}"
            )

    def test_V15_emergency_stop_desktop_and_mobile(self):
        content = self._src_content()
        self.assertIn(
            'id="btn-estop"',
            content,
            "index.html must have desktop emergency stop button #btn-estop"
        )
        self.assertIn(
            'id="btn-estop-m"',
            content,
            "index.html must have mobile emergency stop button #btn-estop-m"
        )

    def test_V16_inline_panel_hidden_rule_precedes_responsive_overrides(self):
        import re

        content = self._src_content()
        hidden_match = re.search(
            r"\.tablet-system-status,\s*"
            r"\.tablet-controls,\s*"
            r"\.tablet-quick-info\s*\{\s*display:\s*none;\s*\}",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(
            hidden_match,
            "desktop inline-panel hidden rule must be present",
        )

        for max_width in ("1199", "767"):
            media_match = re.search(
                rf"@media\s*\(max-width:\s*{max_width}px\)\s*\{{(?P<body>.*?)"
                r"(?=\n\s*@media|\n\s*</style>)",
                content,
                re.DOTALL,
            )
            self.assertIsNotNone(
                media_match,
                f"max-width:{max_width}px responsive block must be present",
            )
            media_body = media_match.group("body")
            for cls in ("tablet-system-status", "tablet-controls", "tablet-quick-info"):
                self.assertRegex(
                    media_body,
                    rf"\.{cls}\s*\{{\s*display:\s*block;\s*\}}",
                    f".{cls} must be shown in max-width:{max_width}px layout",
                )
            self.assertLess(
                hidden_match.start(),
                media_match.start(),
                "desktop display:none rule must come before responsive display:block "
                f"overrides for max-width:{max_width}px",
            )


class TestTodaySummaryUI(unittest.TestCase):
    """
    Today Summary dashboard tests (TS01–TS15).

    TS01. Today Summary section exists in authoritative index.html.
    TS02. Four metric cards exist (stat-inspected, stat-helmet, stat-no-helmet, stat-warnings).
    TS03. UI reads state.statistics.inspected.
    TS04. UI reads state.statistics.helmet.
    TS05. UI reads state.statistics.no_helmet.
    TS06. UI reads state.statistics.warnings.
    TS07. Missing stats value renders em-dash, not zero.
    TS08. Backend zero is rendered as '0' (statVal distinguishes zero from absent).
    TS09. Desktop 4-column today-grid rule exists.
    TS10. Tablet 2x2 today-grid rule exists.
    TS11. Mobile 2x2 today-grid rule exists.
    TS12. today-grid class present in HTML.
    TS13. No duplicated polling loop introduced.
    TS14. stat-warnings card is semantically distinct from no-helmet card.
    TS15. statistics.date is used for Today Summary date display.
    """

    def _src(self):
        return SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")

    def test_TS01_today_summary_section_exists(self):
        content = self._src()
        self.assertIn("TODAY SUMMARY", content,
                      "index.html must contain TODAY SUMMARY section")

    def test_TS02_four_metric_card_ids_exist(self):
        content = self._src()
        for card_id in ("stat-inspected", "stat-helmet", "stat-no-helmet", "stat-warnings"):
            self.assertIn(f'id="{card_id}"', content,
                          f"index.html must have element with id='{card_id}'")

    def test_TS03_ui_reads_statistics_inspected(self):
        content = self._src()
        self.assertIn("stats.inspected", content,
                      "index.html must read state.statistics.inspected")

    def test_TS04_ui_reads_statistics_helmet(self):
        content = self._src()
        self.assertIn("stats.helmet", content,
                      "index.html must read state.statistics.helmet")

    def test_TS05_ui_reads_statistics_no_helmet(self):
        content = self._src()
        self.assertIn("stats.no_helmet", content,
                      "index.html must read state.statistics.no_helmet")

    def test_TS06_ui_reads_statistics_warnings(self):
        content = self._src()
        self.assertIn("stats.warnings", content,
                      "index.html must read state.statistics.warnings")

    def test_TS07_missing_value_renders_emdash_not_zero(self):
        import re
        content = self._src()
        self.assertIn("statVal", content,
                      "index.html must use statVal helper to guard missing values")
        self.assertRegex(
            content,
            r"typeof\s+v\s*===\s*[\"']number[\"']",
            "statVal must check typeof v === 'number' before rendering"
        )

    def test_TS08_backend_zero_renders_as_string_zero(self):
        import re
        content = self._src()
        self.assertRegex(
            content,
            r"String\(v\)",
            "statVal must use String(v) to render numeric zero as '0'"
        )

    def test_TS09_desktop_four_column_today_grid_exists(self):
        import re
        content = self._src()
        self.assertRegex(
            content,
            r"\.today-grid\s*\{[^}]*grid-template-columns:\s*repeat\(\s*4\s*,\s*1fr\s*\)",
            "today-grid must have 4-column rule for desktop"
        )

    def test_TS10_tablet_two_column_today_grid_exists(self):
        import re
        content = self._src()
        tablet_block_match = re.search(
            r"@media\s*\(max-width:\s*1199px\)(.*?)(?=@media|\Z)",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(tablet_block_match, "max-width:1199px block not found")
        tablet_block = tablet_block_match.group(1)
        self.assertIn("today-grid", tablet_block,
                      "today-grid must be overridden in max-width:1199px tablet breakpoint")
        self.assertIn("repeat(2, 1fr)", tablet_block,
                      "today-grid must be 2-column in tablet breakpoint")

    def test_TS11_mobile_two_column_today_grid_exists(self):
        import re
        content = self._src()
        mobile_block_match = re.search(
            r"@media\s*\(max-width:\s*767px\)(.*?)(?=@media|\Z)",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(mobile_block_match, "max-width:767px block not found")
        mobile_block = mobile_block_match.group(1)
        self.assertIn("today-grid", mobile_block,
                      "today-grid must be overridden in max-width:767px mobile breakpoint")

    def test_TS12_today_grid_class_present_in_html(self):
        content = self._src()
        self.assertIn('class="today-grid"', content,
                      "index.html must use class='today-grid' in Today Summary HTML")

    def test_TS13_no_duplicate_polling_loop(self):
        import re
        content = self._src()
        poll_defs = re.findall(r"\bfunction\s+poll\s*\(", content)
        self.assertEqual(len(poll_defs), 1,
                         f"index.html must define poll() exactly once, found {len(poll_defs)}")
        set_intervals = re.findall(r"\bsetInterval\s*\(\s*poll\b", content)
        self.assertEqual(len(set_intervals), 1,
                         f"index.html must call setInterval(poll,...) exactly once, found {len(set_intervals)}")

    def test_TS14_warnings_card_distinct_from_no_helmet_card(self):
        content = self._src()
        self.assertIn("stat-warnings", content,
                      "warnings card must have id='stat-warnings'")
        self.assertIn("stat-no-helmet", content,
                      "no-helmet card must have id='stat-no-helmet'")
        warnings_pos = content.find('id="stat-warnings"')
        no_helmet_pos = content.find('id="stat-no-helmet"')
        self.assertNotEqual(warnings_pos, -1)
        self.assertNotEqual(no_helmet_pos, -1)
        self.assertNotEqual(warnings_pos, no_helmet_pos,
                            "warnings and no-helmet must be separate elements")

    def test_TS15_statistics_date_used_in_today_summary(self):
        content = self._src()
        self.assertIn("stats.date", content,
                      "index.html must use state.statistics.date for Today Summary date")


class TestRecentEventsLayout(unittest.TestCase):
    """
    Recent Events panel layout and KST rollover tests (RE01–RE24).

    RE01.  Recent Events panel exists in authoritative index.html.
    RE02.  event-list element exists for event rendering.
    RE03.  events card body has overflow-y: auto.
    RE04.  events card body has a bounded height (not flex-grow into full screen).
    RE05.  events-card does NOT use flex: 1 (must not expand to fill viewport).
    RE06.  Live Camera uses aspect-ratio: 16 / 9.
    RE07.  Desktop viewport shell uses 100dvh with 100vh fallback.
    RE08.  Desktop body/main does not introduce page scroll via min-height.
    RE09.  Mobile layout restores page scroll (overflow: auto).
    RE10.  Mobile #app does not force fixed viewport height.
    RE11.  Tablet responsive display:block inline panels still present.
    RE12.  Emergency Stop accessible on desktop and mobile.
    RE13.  No horizontal primary page overflow rule.
    RE14.  empty event state still "No events yet".
    RE15.  frontend does not fabricate events.
    RE16.  frontend does not perform daily event reset (no JS date logic for events).
    RE17.  no duplicate setInterval(poll,...) introduced.
    RE18.  Today Summary has four metric cards.
    RE19.  Today Summary KST date reference present.
    RE20.  backend events key consumed directly (no separate event store in JS).
    RE21.  generated assets/index.html reflects updated layout.
    RE22.  app-root assets path remains correct (arduino:web_ui present).
    RE23.  legacy python/assets remains absent.
    RE24.  event list is ordered oldest-first in backend (slice().reverse() in JS).
    """

    def _src(self):
        return SRC_UI_ASSETS.joinpath("index.html").read_text(encoding="utf-8")

    def _gen(self):
        _run_generator()
        return OUT_INDEX_HTML.read_text(encoding="utf-8")

    def test_RE01_recent_events_panel_exists(self):
        content = self._src()
        self.assertIn("Recent Events", content,
                      "index.html must contain Recent Events panel")

    def test_RE02_event_list_element_exists(self):
        content = self._src()
        self.assertIn('id="event-list"', content,
                      "index.html must have #event-list element")

    def test_RE03_events_card_body_has_overflow_y_auto(self):
        content = self._src()
        import re
        events_card_match = re.search(
            r"\.events-card\s+\.card-body\s*\{([^}]+)\}",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(events_card_match,
                             ".events-card .card-body rule must exist")
        body_rule = events_card_match.group(1)
        self.assertIn("overflow-y", body_rule,
                      ".events-card .card-body must set overflow-y")
        self.assertIn("auto", body_rule,
                      ".events-card .card-body overflow-y must be auto")

    def test_RE04_events_card_body_has_bounded_height(self):
        content = self._src()
        import re
        events_card_match = re.search(
            r"\.events-card\s+\.card-body\s*\{([^}]+)\}",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(events_card_match,
                             ".events-card .card-body rule must exist")
        body_rule = events_card_match.group(1)
        self.assertRegex(
            body_rule,
            r"height\s*:",
            ".events-card .card-body must have a bounded height property"
        )
        self.assertNotRegex(
            body_rule,
            r"height\s*:\s*100",
            ".events-card .card-body must not fill 100% height"
        )

    def test_RE05_events_card_does_not_flex_grow(self):
        content = self._src()
        import re
        events_card_base_match = re.search(
            r"\.events-card\s*\{([^}]+)\}",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(events_card_base_match,
                             ".events-card base rule must exist")
        base_rule = events_card_base_match.group(1)
        self.assertNotRegex(
            base_rule,
            r"flex\s*:\s*1",
            ".events-card must not use flex: 1 (prevents consuming full remaining space)"
        )

    def test_RE06_camera_has_aspect_ratio_16_9(self):
        content = self._src()
        self.assertIn("aspect-ratio: 16 / 9", content,
                      ".camera-wrap must preserve aspect-ratio: 16 / 9")

    def test_RE07_desktop_viewport_uses_100dvh_with_100vh_fallback(self):
        content = self._src()
        self.assertIn("100vh", content, "#app must use 100vh as fallback")
        self.assertIn("100dvh", content, "#app must use 100dvh for modern browsers")
        vh_pos = content.find("100vh")
        dvh_pos = content.find("100dvh")
        self.assertLess(vh_pos, dvh_pos,
                        "100vh fallback must appear before 100dvh in source order")

    def test_RE08_main_scroll_does_not_force_min_height_100(self):
        content = self._src()
        import re
        main_scroll_match = re.search(
            r"\.main-scroll\s*\{([^}]+)\}",
            content,
            re.DOTALL,
        )
        if main_scroll_match:
            rule = main_scroll_match.group(1)
            self.assertNotRegex(
                rule,
                r"min-height\s*:\s*100%",
                ".main-scroll must not use min-height:100% (forces page scroll on desktop)"
            )

    def test_RE09_mobile_layout_restores_page_scroll(self):
        content = self._src()
        import re
        mobile_block_match = re.search(
            r"@media\s*\(max-width:\s*767px\)(.*?)(?=@media|\Z)",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(mobile_block_match, "max-width:767px block not found")
        mobile_block = mobile_block_match.group(1)
        self.assertRegex(
            mobile_block,
            r"overflow\s*:\s*auto",
            "Mobile breakpoint must restore overflow: auto for page scroll"
        )

    def test_RE10_mobile_app_does_not_force_fixed_height(self):
        content = self._src()
        import re
        mobile_block_match = re.search(
            r"@media\s*\(max-width:\s*767px\)(.*?)(?=@media|\Z)",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(mobile_block_match, "max-width:767px block not found")
        mobile_block = mobile_block_match.group(1)
        self.assertRegex(
            mobile_block,
            r"#app\s*\{[^}]*height\s*:\s*auto",
            "Mobile #app must use height:auto to allow page scroll"
        )

    def test_RE11_tablet_inline_panels_still_displayed(self):
        content = self._src()
        import re
        tablet_block_match = re.search(
            r"@media\s*\(max-width:\s*1199px\)(.*?)(?=@media|\Z)",
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(tablet_block_match, "max-width:1199px block not found")
        tablet_block = tablet_block_match.group(1)
        for cls in ("tablet-system-status", "tablet-controls", "tablet-quick-info"):
            self.assertRegex(
                tablet_block,
                rf"\.{cls}\s*\{{\s*display:\s*block;\s*\}}",
                f".{cls} must be visible in tablet layout"
            )

    def test_RE12_emergency_stop_accessible_desktop_and_mobile(self):
        content = self._src()
        self.assertIn('id="btn-estop"', content,
                      "Desktop emergency stop must exist")
        self.assertIn('id="btn-estop-m"', content,
                      "Mobile emergency stop must exist")

    def test_RE13_no_primary_horizontal_overflow_introduced(self):
        content = self._src()
        self.assertIn("overflow-x: hidden", content,
                      "Main area must prevent horizontal overflow")
        self.assertNotIn("overflow-x: scroll", content,
                         "index.html must not introduce horizontal scroll")

    def test_RE14_empty_event_state_shows_no_events_yet(self):
        content = self._src()
        self.assertIn("No events yet", content,
                      "Empty event state must show 'No events yet'")

    def test_RE15_frontend_does_not_fabricate_events(self):
        content = self._src()
        fabrication_indicators = [
            "fake_event",
            "mock_event",
            "sample_event",
            "pushEvent(",
            "events.push(",
        ]
        for indicator in fabrication_indicators:
            self.assertNotIn(indicator, content,
                             f"Frontend must not fabricate events: '{indicator}'")

    def test_RE16_frontend_does_not_perform_daily_reset(self):
        content = self._src()
        reset_indicators = [
            "midnight",
            "events = []",
            "events.length = 0",
            "clearEvents",
            "resetEvents",
        ]
        for indicator in reset_indicators:
            self.assertNotIn(indicator, content,
                             f"Frontend must not perform daily event reset: '{indicator}'")

    def test_RE17_no_duplicate_poll_interval(self):
        import re
        content = self._src()
        set_intervals = re.findall(r"\bsetInterval\s*\(\s*poll\b", content)
        self.assertEqual(len(set_intervals), 1,
                         "setInterval(poll,...) must appear exactly once")

    def test_RE18_today_summary_has_four_metric_cards(self):
        content = self._src()
        for card_id in ("stat-inspected", "stat-helmet", "stat-no-helmet", "stat-warnings"):
            self.assertIn(f'id="{card_id}"', content,
                          f"Today Summary must have metric card: {card_id}")

    def test_RE19_today_summary_kst_date_referenced(self):
        content = self._src()
        self.assertIn("stats.date", content,
                      "Today Summary must reference state.statistics.date")

    def test_RE20_events_consumed_from_api_state(self):
        content = self._src()
        self.assertIn("snap.events", content,
                      "Frontend must consume events from /api/state snapshot")

    def test_RE21_generated_assets_reflect_updated_layout(self):
        content = self._gen()
        self.assertIn("overflow-y", content,
                      "Generated assets/index.html must contain overflow-y for events")
        self.assertIn("Recent Events", content,
                      "Generated assets/index.html must contain Recent Events panel")

    def test_RE22_arduino_web_ui_configured(self):
        _run_generator()
        content = APP_YAML.read_text(encoding="utf-8")
        self.assertIn("arduino:web_ui", content,
                      "app.yaml must declare arduino:web_ui")

    def test_RE23_legacy_python_assets_absent(self):
        _run_generator()
        self.assertFalse(OLD_OUT_INDEX_HTML.exists(),
                         "Legacy python/assets/index.html must not exist")

    def test_RE24_frontend_reverses_events_for_newest_first(self):
        content = self._src()
        self.assertIn(".reverse()", content,
                      "Frontend must reverse event list so newest events appear first")


if __name__ == "__main__":
    unittest.main()
