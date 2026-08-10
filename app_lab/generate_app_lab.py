"""
App Lab Package Generator
=========================
Generates a self-contained Arduino App Lab deployment package from authoritative sources.

Authoritative sources (never modified by this script):
    mpu/        – Python runtime package
    arduino/    – MCU firmware source
    app_lab/ui/ – Dashboard web assets (HTML/CSS/JS templates)

Generated output (always overwritten, never manually maintained):
    app_lab/Arduino Safety Monitor/python/mpu/         – copied from mpu/
    app_lab/Arduino Safety Monitor/python/assets/      – copied from app_lab/ui/assets/
    app_lab/Arduino Safety Monitor/sketch/             – copied from arduino/

Usage (run from repository root):
    python app_lab/generate_app_lab.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

AUTH_MPU = REPO_ROOT / "mpu"
AUTH_ARDUINO = REPO_ROOT / "arduino"
AUTH_INO = AUTH_ARDUINO / "arduino.ino"

APP_LAB_DIR = REPO_ROOT / "app_lab"
APP_DIR = APP_LAB_DIR / "Arduino Safety Monitor"
OUT_PYTHON_DIR = APP_DIR / "python"
OUT_MPU_DIR = OUT_PYTHON_DIR / "mpu"
OUT_ASSETS_DIR = OUT_PYTHON_DIR / "assets"
OUT_SKETCH_DIR = APP_DIR / "sketch"
OUT_ADAPTER = OUT_PYTHON_DIR / "main.py"
OUT_APP_YAML = APP_DIR / "app.yaml"
OUT_SKETCH_YAML = OUT_SKETCH_DIR / "sketch.yaml"
OUT_SKETCH_INO = OUT_SKETCH_DIR / "sketch.ino"
OUT_MARKER = APP_DIR / "GENERATED_FROM_SOURCE.md"
OUT_REQUIREMENTS = OUT_PYTHON_DIR / "requirements.txt"

SRC_REQUIREMENTS = APP_LAB_DIR / "requirements_app_lab.txt"
SRC_UI_ASSETS = APP_LAB_DIR / "ui" / "assets"

_EXCLUDE_DIRS = {"__pycache__", "tests"}
_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
_EXCLUDE_MPU_FILES = {"ai/train.py", "ai/convert.py", "ai/dataset/loader.py"}


def _fail(msg: str) -> None:
    print(f"[generate_app_lab] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _log(msg: str) -> None:
    print(f"[generate_app_lab] {msg}")


def _check_authoritative_sources() -> None:
    if not AUTH_MPU.is_dir():
        _fail(f"Authoritative mpu/ not found at: {AUTH_MPU}")
    if not AUTH_ARDUINO.is_dir():
        _fail(f"Authoritative arduino/ not found at: {AUTH_ARDUINO}")
    if not AUTH_INO.is_file():
        _fail(f"Authoritative arduino/arduino.ino not found at: {AUTH_INO}")
    if not SRC_REQUIREMENTS.is_file():
        _fail(f"App Lab dependency source not found at: {SRC_REQUIREMENTS}")
    if not SRC_UI_ASSETS.is_dir():
        _fail(f"UI assets source not found at: {SRC_UI_ASSETS}")
    _log("Authoritative sources verified.")


def _clear_generated_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        _log(f"Cleared stale generated tree: {path.relative_to(REPO_ROOT)}")


_FORBIDDEN_DELETION_TARGETS = (REPO_ROOT, AUTH_MPU, AUTH_ARDUINO, APP_LAB_DIR)


def _clear_app_dir() -> None:
    target = APP_DIR.resolve()
    if not str(target).startswith(str(APP_LAB_DIR.resolve()) + "/"):
        _fail(
            f"Path safety violation: APP_DIR ({target}) is not strictly under "
            f"APP_LAB_DIR ({APP_LAB_DIR.resolve()}).  Refusing to delete."
        )
    for forbidden in _FORBIDDEN_DELETION_TARGETS:
        if target == forbidden.resolve():
            _fail(
                f"Path safety violation: APP_DIR ({target}) equals a protected "
                f"directory ({forbidden}).  Refusing to delete."
            )
    _clear_generated_tree(APP_DIR)


def _copy_mpu_package() -> None:
    OUT_MPU_DIR.mkdir(parents=True, exist_ok=True)

    _exclude_rel = {str(Path(p)) for p in _EXCLUDE_MPU_FILES}

    for src in sorted(AUTH_MPU.rglob("*")):
        rel = src.relative_to(AUTH_MPU)
        rel_str = str(rel)

        if any(part in _EXCLUDE_DIRS for part in rel.parts):
            continue
        if src.suffix in _EXCLUDE_SUFFIXES:
            continue
        if rel_str in _exclude_rel:
            continue

        dst = OUT_MPU_DIR / rel

        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    _log(f"Copied mpu/ -> {OUT_MPU_DIR.relative_to(REPO_ROOT)}")


def _copy_ui_assets() -> None:
    OUT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(SRC_UI_ASSETS.rglob("*")):
        if src.suffix in _EXCLUDE_SUFFIXES:
            continue
        rel = src.relative_to(SRC_UI_ASSETS)
        dst = OUT_ASSETS_DIR / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    _log(f"Copied ui/assets/ -> {OUT_ASSETS_DIR.relative_to(REPO_ROOT)}")


def _discover_arduino_headers() -> list[Path]:
    headers = []
    for p in sorted(AUTH_ARDUINO.iterdir()):
        if p.is_file() and p.suffix in {".h", ".hpp", ".cpp"}:
            headers.append(p)
    return headers


def _copy_sketch() -> None:
    OUT_SKETCH_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(AUTH_INO, OUT_SKETCH_INO)
    _log(f"Copied arduino.ino -> sketch/sketch.ino")

    headers = _discover_arduino_headers()
    for hdr in headers:
        shutil.copy2(hdr, OUT_SKETCH_DIR / hdr.name)
        _log(f"Copied arduino/{hdr.name} -> sketch/{hdr.name}")


def _write_app_yaml() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    content = (
        "name: Arduino Safety Monitor\n"
        "icon: \U0001F60E\n"
        "ports: []\n"
        "bricks:\n"
        "  - arduino:web_ui: {}\n"
    )
    OUT_APP_YAML.write_text(content, encoding="utf-8")
    _log(f"Written {OUT_APP_YAML.relative_to(REPO_ROOT)}")


def _write_sketch_yaml() -> None:
    content = (
        "profiles:\n"
        "  default:\n"
        "    platforms:\n"
        "      - platform: arduino:zephyr\n"
        "    libraries:\n"
        "default_profile: default\n"
    )
    OUT_SKETCH_YAML.write_text(content, encoding="utf-8")
    _log(f"Written {OUT_SKETCH_YAML.relative_to(REPO_ROOT)}")


def _write_adapter() -> None:
    lines = [
        '"""',
        "App Lab Python Adapter \u2013 Arduino Safety Monitor",
        "================================================",
        "GENERATED FILE: DO NOT EDIT.",
        "Edit authoritative source: mpu/main.py and app_lab/generate_app_lab.py",
        "",
        "Lifecycle design:",
        "    HelmetDetectionSystem.start() owns a blocking camera + AI loop.",
        "    App.run(user_loop=...) owns its own event loop.",
        "    Both cannot run on the same thread.",
        "",
        "    Strategy:",
        "      1. HelmetDetectionSystem is constructed once at import time (deferred",
        "         to user_loop first call to avoid constructor side-effects at import).",
        "      2. On the first user_loop tick the system is constructed and its blocking",
        "         start() is launched in a daemon thread.",
        "      3. user_loop() monitors the thread and HelmetDetectionSystem.running each",
        "         tick, logging state changes.",
        "      4. Dashboard API endpoints are registered on the WebUI brick before",
        "         App.run() to make them available from the first request.",
        "",
        "Hardware-free development mode:",
        "    Set APP_LAB_DEV_MODE=true (or 1 / yes) in the container environment to",
        "    run without USB camera or Arduino serial.  Camera / serial failures are",
        "    logged as warnings instead of raising RuntimeError.  Default is strict",
        "    (production) mode: hardware absence is treated as an error.",
        "",
        "Shutdown limitation:",
        "    The arduino.app_utils.App API as confirmed from real usage provides",
        "    App.run(user_loop=fn).  No documented shutdown hook was available for",
        "    inspection in this environment.  Therefore HelmetDetectionSystem.stop() is",
        "    registered via atexit as the best-effort orderly shutdown mechanism.",
        "    On normal App exit the atexit handler fires and stop() executes.",
        "    On hard kill (SIGKILL) cleanup may not occur; this matches behaviour of",
        "    the production mpu/main.py entry point.",
        '"""',
        "",
        "import atexit",
        "import logging",
        "import os",
        "import threading",
        "",
        "# App Lab deployment packages default to hardware-free development mode.",
        "# External APP_LAB_DEV_MODE=false is respected for later real-hardware tests.",
        'os.environ.setdefault("APP_LAB_DEV_MODE", "true")',
        "",
        "from arduino.app_utils import App",
        "from arduino.app_bricks.web_ui import WebUI",
        "from mpu.main import HelmetDetectionSystem",
        "from mpu.config import (",
        "    DEFAULT_SERIAL_PORT, DEFAULT_SERVER_URL, APP_LAB_DEV_MODE,",
        "    validate_runtime_models,",
        ")",
        "from mpu.ui_state import build_ui_payload",
        "",
        "logging.basicConfig(level=logging.INFO)",
        "_logger = logging.getLogger(__name__)",
        "",
        "_system: HelmetDetectionSystem | None = None",
        "_worker_thread: threading.Thread | None = None",
        "_started = False",
        "",
        "_ui = WebUI()",
        "",
        "",
        "def _shutdown_hook() -> None:",
        "    global _system",
        "    if _system is not None:",
        '        _logger.info("App Lab adapter: atexit shutdown \u2013 calling HelmetDetectionSystem.stop()")',
        "        try:",
        "            _system.stop()",
        "        except Exception as exc:",
        '            _logger.warning("stop() raised during atexit: %s", exc)',
        "",
        "",
        "atexit.register(_shutdown_hook)",
        "",
        "",
        "def _api_state():",
        "    if _system is not None and hasattr(_system, \"dashboard\"):",
        "        snap = _system.dashboard.snapshot()",
        "        camera_available = getattr(",
        "            getattr(_system, \"camera\", None), \"_camera_available\", False",
        "        )",
        "    else:",
        "        from mpu.dashboard_state import DashboardState",
        "        snap = DashboardState().snapshot()",
        "        camera_available = False",
        "    return build_ui_payload(",
        "        snap,",
        "        dev_mode=APP_LAB_DEV_MODE,",
        "        camera_available=camera_available,",
        "        serial_port=DEFAULT_SERIAL_PORT,",
        "        server_url=DEFAULT_SERVER_URL,",
        "    )",
        "",
        "",
        "def _api_stop():",
        "    if _system is None or not getattr(_system, \"_connected\", False):",
        "        return {\"error\": \"Hardware not connected\"}",
        "    try:",
        "        _system.bridge_rpc.motor_control(\"stop\")",
        "        return {\"ok\": True}",
        "    except Exception as exc:",
        '        _logger.warning("stop command failed: %s", exc)',
        "        return {\"error\": str(exc)}",
        "",
        "",
        "def _api_reset():",
        "    if _system is None or not getattr(_system, \"_connected\", False):",
        "        return {\"error\": \"Hardware not connected\"}",
        "    try:",
        "        _system.bridge_rpc.safe_reset()",
        "        return {\"ok\": True}",
        "    except Exception as exc:",
        '        _logger.warning("reset command failed: %s", exc)',
        "        return {\"error\": str(exc)}",
        "",
        "",
        "def _api_estop():",
        "    if _system is None or not getattr(_system, \"_connected\", False):",
        "        return {\"error\": \"Hardware not connected\"}",
        "    try:",
        "        _system.bridge_rpc.motor_control(\"stop\")",
        "        _system.bridge_rpc.safe_reset()",
        "        return {\"ok\": True}",
        "    except Exception as exc:",
        '        _logger.warning("emergency stop failed: %s", exc)',
        "        return {\"error\": str(exc)}",
        "",
        "",
        "def _api_test_led():",
        "    if _system is None or not getattr(_system, \"_connected\", False):",
        "        return {\"error\": \"Hardware not connected\"}",
        "    try:",
        "        _system.bridge_rpc.led_control(\"red\")",
        "        return {\"ok\": True}",
        "    except Exception as exc:",
        "        return {\"error\": str(exc)}",
        "",
        "",
        "def _api_test_buzzer():",
        "    return {\"error\": \"unsupported: no safe buzzer-off lifecycle in dashboard\"}",
        "",
        "",
        '_ui.expose_api("GET",  "/api/state",           _api_state)',
        '_ui.expose_api("POST", "/api/control/stop",    _api_stop)',
        '_ui.expose_api("POST", "/api/control/reset",   _api_reset)',
        '_ui.expose_api("POST", "/api/control/estop",   _api_estop)',
        '_ui.expose_api("POST", "/api/control/test_led",    _api_test_led)',
        '_ui.expose_api("POST", "/api/control/test_buzzer", _api_test_buzzer)',
        "",
        "",
        "def user_loop() -> None:",
        "    global _system, _worker_thread, _started",
        "",
        "    if not _started:",
        "        _started = True",
        '        _logger.info("App Lab adapter: initialising HelmetDetectionSystem")',
        "        try:",
        "            validate_runtime_models()",
        "        except FileNotFoundError as exc:",
        '            _logger.error("Model files missing \u2013 system will not start: %s", exc)',
        "            return",
        "",
        "        _system = HelmetDetectionSystem(",
        "            port=DEFAULT_SERIAL_PORT,",
        "            server_url=DEFAULT_SERVER_URL,",
        "        )",
        "        _worker_thread = threading.Thread(",
        "            target=_system.start,",
        '            name="helmet-detection-worker",',
        "            daemon=True,",
        "        )",
        "        _worker_thread.start()",
        '        _logger.info("App Lab adapter: HelmetDetectionSystem worker thread started")',
        "        return",
        "",
        "    if _system is not None and not _system.running:",
        '        _logger.warning("App Lab adapter: HelmetDetectionSystem.running is False")',
        "",
        "    if _worker_thread is not None and not _worker_thread.is_alive():",
        '        _logger.warning("App Lab adapter: worker thread has exited")',
        "",
        "",
        "App.run(user_loop=user_loop)",
    ]
    content = "\n".join(lines) + "\n"
    OUT_ADAPTER.write_text(content, encoding="utf-8")
    _log(f"Written {OUT_ADAPTER.relative_to(REPO_ROOT)}")


def _write_requirements() -> None:
    OUT_PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    content = SRC_REQUIREMENTS.read_text(encoding="utf-8")
    OUT_REQUIREMENTS.write_text(content, encoding="utf-8")
    _log(f"Written {OUT_REQUIREMENTS.relative_to(REPO_ROOT)}")


def _write_marker() -> None:
    content = (
        "# GENERATED RUNTIME COPIES \u2014 DO NOT EDIT\n\n"
        "The files under this directory tree are **generated** from authoritative sources.\n\n"
        "## Authoritative sources\n\n"
        "| Generated path | Authoritative source |\n"
        "| --- | --- |\n"
        "| `python/mpu/` | `mpu/` (repository root) |\n"
        "| `python/assets/` | `app_lab/ui/assets/` |\n"
        "| `python/requirements.txt` | `app_lab/requirements_app_lab.txt` |\n"
        "| `sketch/sketch.ino` | `arduino/arduino.ino` |\n"
        "| `sketch/*.h` | `arduino/*.h` |\n\n"
        "## To update generated files\n\n"
        "Run from the repository root:\n\n"
        "```\n"
        "python app_lab/generate_app_lab.py\n"
        "```\n\n"
        "**Never edit `python/requirements.txt` directly.**\n"
        "Edit `app_lab/requirements_app_lab.txt` and regenerate.\n\n"
        "**Never edit files inside `Arduino Safety Monitor/python/mpu/`, "
        "`Arduino Safety Monitor/python/assets/`, "
        "or `Arduino Safety Monitor/sketch/` directly.**\n"
        "All edits must be made in `mpu/`, `app_lab/ui/`, or `arduino/` and regenerated.\n"
    )
    OUT_MARKER.write_text(content, encoding="utf-8")
    _log(f"Written {OUT_MARKER.relative_to(REPO_ROOT)}")


def generate() -> None:
    _log("=== App Lab Package Generator ===")
    _check_authoritative_sources()
    _clear_app_dir()

    _write_app_yaml()
    _copy_mpu_package()
    _copy_ui_assets()
    _write_adapter()
    _write_requirements()
    _copy_sketch()
    _write_sketch_yaml()
    _write_marker()

    _log("=== Generation complete ===")
    _log(f"Deploy folder: {APP_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    generate()
