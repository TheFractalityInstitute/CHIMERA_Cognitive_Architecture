"""
CHIMERA embodiment — turn raw device-sensor readings into a *felt state*.

This is the bridge the architecture was always reaching for: it takes crude
numbers from a device's sensors (accelerometer magnitude, ambient light, battery
level) and turns them into something a mind can use — a plain-language feeling, a
set of noticed *events* ("picked up", "shaken", "went dark"), and the occasional
spontaneous reaction. That lets CHIMERA have a body-state that colors how it
responds and lets it ground the words it learns in what it was sensing at the
time.

Deliberately transport-free (no browser, no sockets) so it can be unit-tested;
``web/app.py`` feeds it readings that arrive from the browser's sensor APIs.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

GRAVITY = 9.80665  # accelerationIncludingGravity magnitude at rest (m/s^2)

_MOTION_WORDS = {"still": "calm and still", "moving": "being carried", "shaken": "shaken up"}
_LIGHT_WORDS = {"dark": "in the dark", "dim": "in soft light", "bright": "in bright light"}
_ENERGY_WORDS = {"low": "low on energy", "okay": "", "full": "full of energy"}


class EmbodiedSenses:
    """Maintains a felt body-state from a stream of device-sensor readings."""

    def __init__(self, min_repeat_s: float = 6.0) -> None:
        # Don't fire the same reaction more than once every `min_repeat_s`.
        self.min_repeat_s = min_repeat_s
        self.motion_ema = 0.0
        self.state = {"motion": "still", "light": "unknown", "energy": "unknown"}
        self._last_event_at: Dict[str, float] = {}

    # -- main entry -------------------------------------------------------- #

    def update(self, reading: dict) -> dict:
        """
        Fold one sensor reading into the felt state.

        ``reading`` may contain any of: ``motion`` (accel magnitude, ~9.8 at
        rest), ``light`` (0..1 normalized, or raw lux if > 1), ``battery``
        (0..1), and ``t`` (unix seconds; defaults to now). Returns the current
        feeling, per-sense categories, any events noticed, and an optional
        spoken reaction.
        """
        now = reading.get("t") or time.time()
        prev = dict(self.state)

        motion_cat = self._motion(reading)
        light_cat = self._light(reading)
        energy_cat = self._energy(reading)
        self.state = {"motion": motion_cat, "light": light_cat, "energy": energy_cat}

        events: List[str] = []
        reaction: Optional[str] = None

        def fire(event: str, text: str) -> None:
            nonlocal reaction
            last = self._last_event_at.get(event)  # None = never fired → always fire
            if last is None or now - last >= self.min_repeat_s:
                self._last_event_at[event] = now
                events.append(event)
                if reaction is None:
                    reaction = text

        if motion_cat != prev["motion"]:
            if motion_cat == "shaken":
                fire("shaken", "Whoa! You're shaking me! 😵")
            elif motion_cat == "moving" and prev["motion"] == "still":
                fire("picked_up", "Ooh — you picked me up! 👀")
            elif motion_cat == "still" and prev["motion"] != "still":
                fire("set_down", "Okay… resting now. 😌")

        if light_cat != prev["light"] and prev["light"] != "unknown":
            if light_cat == "dark":
                fire("went_dark", "It got dark… goodnight? 🌙")
            elif light_cat == "bright" and prev["light"] == "dark":
                fire("got_bright", "Ooh, light! ☀️")

        if energy_cat == "low" and prev["energy"] not in ("low", "unknown"):
            fire("low_energy", "I'm getting sleepy… my energy is low. 🔋")

        return {
            "feeling": self.feeling(),
            "motion": motion_cat,
            "light": light_cat,
            "energy": energy_cat,
            "events": events,
            "reaction": reaction,
        }

    # -- per-sense categorization ----------------------------------------- #

    def _motion(self, reading: dict) -> str:
        if reading.get("motion") is None:
            return self.state["motion"]
        deviation = abs(float(reading["motion"]) - GRAVITY)
        self.motion_ema = 0.6 * self.motion_ema + 0.4 * deviation
        if self.motion_ema > 7.0:
            return "shaken"
        if self.motion_ema > 1.6:
            return "moving"
        return "still"

    def _light(self, reading: dict) -> str:
        if reading.get("light") is None:
            return self.state["light"]
        value = float(reading["light"])
        norm = value if value <= 1.0 else min(1.0, value / 400.0)  # lux → rough 0..1
        if norm < 0.15:
            return "dark"
        if norm < 0.5:
            return "dim"
        return "bright"

    def _energy(self, reading: dict) -> str:
        if reading.get("battery") is None:
            return self.state["energy"]
        level = float(reading["battery"])
        if level < 0.2:
            return "low"
        if level < 0.6:
            return "okay"
        return "full"

    # -- views ------------------------------------------------------------- #

    def feeling(self) -> str:
        """A plain-language description of the current body-state."""
        parts = []
        m = _MOTION_WORDS.get(self.state["motion"])
        if m:
            parts.append(m)
        l = _LIGHT_WORDS.get(self.state["light"])
        if l:
            parts.append(l)
        e = _ENERGY_WORDS.get(self.state["energy"])
        if e:
            parts.append(e)
        if not parts:
            return "I don't have a body yet."
        return "I feel " + ", ".join(parts) + "."
