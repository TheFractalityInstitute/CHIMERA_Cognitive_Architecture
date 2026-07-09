# chimera_core/sensors/__init__.py
"""
CHIMERA sensors — device embodiment.

The working piece is ``embodiment.py`` (``EmbodiedSenses``), which turns raw
device-sensor readings into a felt body-state. Import it directly::

    from chimera_core.sensors.embodiment import EmbodiedSenses

The other modules here (``mobile_sensors.py``, ``biometric_sensors.py``) target
native Android/Termux and Garmin hardware and reference an earlier agent
interface; they are reference-only for now. This package root imports nothing
heavy.
"""

__all__ = ["embodiment"]
