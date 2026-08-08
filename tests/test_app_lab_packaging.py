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

APP_DIR = REPO_ROOT / "app_lab" / "Arduino Safety Monitor"
PYTHON_DIR = APP_DIR / "python"
MPU_PACKAGE_DIR = PYTHON_DIR / "mpu"
SKETCH_DIR = APP_DIR / "sketch"
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


if __name__ == "__main__":
    unittest.main()
