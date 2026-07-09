"""
CHIMERA native sensors via Termux:API.

When CHIMERA runs directly on an Android phone inside Termux, this module reads
the device's real hardware — accelerometer, ambient light, battery — by shelling
out to the ``termux-*`` command-line tools (from the Termux:API app). It turns
each reading into the same little dict that ``EmbodiedSenses.update()`` expects,
so the phone's actual body feeds straight into CHIMERA's felt state.

This replaces the older ``mobile_sensors.py`` (which imported a package that
never existed). It degrades gracefully: off a phone, ``available()`` is False and
the readers return ``None`` instead of raising, so the app runs fine anywhere.

Parsing is split from the subprocess calls so it can be unit-tested without a
device.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Pure parsers (testable without a phone)
# --------------------------------------------------------------------------- #


def parse_accelerometer(data: Optional[dict]) -> Optional[float]:
    """Magnitude (m/s^2) from a `termux-sensor -s accelerometer` payload."""
    if not isinstance(data, dict):
        return None
    for entry in data.values():
        values = entry.get("values") if isinstance(entry, dict) else None
        if values and len(values) >= 3:
            x, y, z = values[0], values[1], values[2]
            return math.sqrt(x * x + y * y + z * z)
    return None


def parse_light(data: Optional[dict]) -> Optional[float]:
    """Illuminance (lux) from a `termux-sensor -s light` payload."""
    if not isinstance(data, dict):
        return None
    for entry in data.values():
        values = entry.get("values") if isinstance(entry, dict) else None
        if values:
            return float(values[0])
    return None


def parse_battery(data: Optional[dict]) -> Optional[float]:
    """Battery level 0..1 from a `termux-battery-status` payload."""
    if not isinstance(data, dict):
        return None
    pct = data.get("percentage")
    return pct / 100.0 if isinstance(pct, (int, float)) else None


# --------------------------------------------------------------------------- #
# Device access
# --------------------------------------------------------------------------- #


def available() -> bool:
    """True if the Termux:API sensor tools are installed (i.e. running on a phone)."""
    return shutil.which("termux-sensor") is not None


def _run(cmd: list[str], timeout: float = 4.0) -> Optional[dict]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = result.stdout.strip()
        if result.returncode != 0 or not out:
            return None
        return json.loads(out)
    except Exception:
        # Timeout, bad JSON, missing binary, permission denied — all non-fatal.
        return None


def read_reading() -> dict:
    """
    One combined sensor reading suitable for ``EmbodiedSenses.update()``.

    Always includes ``t`` (timestamp); includes ``motion``/``light``/``battery``
    only for sensors that answered. Returns just ``{"t": ...}`` off a device.
    """
    reading: dict = {"t": time.time()}

    motion = parse_accelerometer(_run(["termux-sensor", "-s", "accelerometer", "-n", "1"]))
    if motion is not None:
        reading["motion"] = motion

    light = parse_light(_run(["termux-sensor", "-s", "light", "-n", "1"]))
    if light is not None:
        reading["light"] = light

    battery = parse_battery(_run(["termux-battery-status"]))
    if battery is not None:
        reading["battery"] = battery

    return reading


def stream(callback: Callable[[dict], None], interval: float = 1.5,
           stop: Optional[Callable[[], bool]] = None) -> None:
    """
    Poll the device's sensors forever, calling ``callback(reading)`` each cycle
    when there's something to report. Blocks — run it in a background thread.
    ``stop`` is an optional predicate; when it returns True the loop ends.
    """
    while not (stop and stop()):
        reading = read_reading()
        if len(reading) > 1:  # more than just the timestamp
            try:
                callback(reading)
            except Exception:
                pass
        time.sleep(interval)
