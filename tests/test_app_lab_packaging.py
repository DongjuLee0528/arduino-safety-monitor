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

APP_DIR = REPO_ROOT / "app_lab" / "Arduino Safety Monitor"
PYTHON_DIR = APP_DIR / "python"
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


if __name__ == "__main__":
    unittest.main()
