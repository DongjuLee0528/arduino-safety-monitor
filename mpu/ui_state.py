"""
UI State Bridge – mpu/ui_state.py
==================================
Converts the authoritative DashboardState snapshot into a JSON-serialisable
view-model dict consumed by the App Lab web dashboard.

Responsibilities (SRP):
    - Read DashboardState.snapshot() (no mutation).
    - Attach non-sensitive runtime info (mode, camera availability, port label).
    - Strip secrets and environment dumps.
    - Return a plain dict safe for JSON serialisation.

This module does NOT:
    - Drive any hardware.
    - Import CameraCapture, BridgeRPC, or HelmetDetectionSystem directly.
    - Expose environment secrets (tokens, passwords, full env dump).
"""

from __future__ import annotations

from typing import Optional

VERSION = "0.1.0"

_REDACTED = "***"
_SAFE_SERVER_PREFIXES = ("http://", "https://")


def _safe_server_url(url: Optional[str]) -> str:
    """Return url with any embedded credentials redacted, or '---' if None."""
    if not url:
        return "---"
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            safe = parsed._replace(netloc=f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname)
            return urlunparse(safe)
    except Exception:
        pass
    return url


def build_ui_payload(
    dashboard_snapshot: dict,
    *,
    dev_mode: bool,
    camera_available: bool,
    serial_port: str,
    server_url: str,
    warning_active: Optional[bool] = None,
) -> dict:
    """
    Build the JSON payload delivered to the dashboard frontend.

    Args:
        dashboard_snapshot: Result of DashboardState.snapshot() — a plain dict.
        dev_mode:           True when APP_LAB_DEV_MODE is active.
        camera_available:   True when CameraCapture opened successfully.
        serial_port:        Configured serial port label (e.g. '/dev/ttyUSB0').
        server_url:         Alert server URL; embedded credentials are redacted.
        warning_active:     True when the D3 LED helmet-warning window is open,
                            False when authoritatively known to be off,
                            None when state is not yet known.

    Returns:
        Plain dict, JSON-serialisable.  Keys:
            state         – full DashboardState snapshot
            dev_mode      – bool
            warning_active– bool or None (None means unknown)
            info          – non-sensitive runtime metadata
    """
    return {
        "state": dashboard_snapshot,
        "dev_mode": bool(dev_mode),
        "warning_active": warning_active,
        "info": {
            "camera_available": bool(camera_available),
            "serial_port": serial_port or "---",
            "server_url": _safe_server_url(server_url),
            "version": VERSION,
        },
    }
