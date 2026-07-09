"""
Tests for the Termux native-sensor parsers.

These exercise the pure parsing logic with sample `termux-*` JSON payloads, so
they run anywhere (no Android device needed). The subprocess calls themselves are
only exercised on a real phone.
"""

import math

from chimera_core.sensors import termux_sensors as ts


def test_parse_accelerometer_magnitude():
    payload = {"LSM6DSO Accelerometer": {"values": [0.0, 0.0, 9.81]}}
    assert math.isclose(ts.parse_accelerometer(payload), 9.81, rel_tol=1e-3)


def test_parse_accelerometer_handles_odd_sensor_name():
    payload = {"Some Vendor 3-axis Accel": {"values": [3.0, 4.0, 0.0]}}
    assert math.isclose(ts.parse_accelerometer(payload), 5.0, rel_tol=1e-6)


def test_parse_light_lux():
    payload = {"TCS3701 Light": {"values": [412.0]}}
    assert ts.parse_light(payload) == 412.0


def test_parse_battery_normalizes_to_fraction():
    payload = {"percentage": 85, "status": "DISCHARGING"}
    assert ts.parse_battery(payload) == 0.85


def test_parsers_tolerate_garbage():
    for bad in (None, {}, {"x": {}}, {"x": {"values": []}}, "nonsense"):
        assert ts.parse_accelerometer(bad) is None
        assert ts.parse_light(bad) is None
    assert ts.parse_battery(None) is None
    assert ts.parse_battery({"percentage": None}) is None


def test_available_is_false_without_termux():
    # This test suite does not run inside Termux, so the CLI must be absent.
    assert ts.available() is False


def test_read_reading_off_device_is_just_timestamp():
    reading = ts.read_reading()
    assert set(reading) == {"t"}  # no sensors answered → only the timestamp
