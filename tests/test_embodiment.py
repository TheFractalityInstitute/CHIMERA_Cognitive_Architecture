"""
Tests for EmbodiedSenses — device sensations → felt state, events, reactions.
"""

from chimera_core.sensors.embodiment import EmbodiedSenses, GRAVITY


def test_rest_feels_still():
    s = EmbodiedSenses()
    out = s.update({"motion": GRAVITY, "t": 0})
    assert out["motion"] == "still"
    assert "still" in out["feeling"]


def test_pickup_then_shake_produces_reactions():
    s = EmbodiedSenses(min_repeat_s=0)
    s.update({"motion": GRAVITY, "t": 0})  # at rest

    # Reactions fire on the *transition* update, so aggregate across the stream.
    events, reactions = [], []

    def stream(magnitude, times):
        for t in times:
            out = s.update({"motion": magnitude, "t": t})
            events.extend(out["events"])
            if out["reaction"]:
                reactions.append(out["reaction"])

    stream(GRAVITY + 3.5, range(1, 6))    # gentle → picked up
    assert s.state["motion"] == "moving"
    assert "picked_up" in events

    stream(GRAVITY + 25, range(6, 12))    # violent → shaken
    assert s.state["motion"] == "shaken"
    assert "shaken" in events
    assert reactions  # at least one spoken reaction happened


def test_going_dark_reacts():
    s = EmbodiedSenses(min_repeat_s=0)
    s.update({"light": 0.9, "t": 0})   # bright
    out = s.update({"light": 0.02, "t": 1})  # dark
    assert out["light"] == "dark"
    assert "went_dark" in out["events"]
    assert "dark" in out["reaction"].lower()


def test_low_battery_is_low_energy():
    s = EmbodiedSenses(min_repeat_s=0)
    s.update({"battery": 0.9, "t": 0})
    out = s.update({"battery": 0.1, "t": 1})
    assert out["energy"] == "low"
    assert "low_energy" in out["events"]


def test_lux_values_are_normalized():
    s = EmbodiedSenses()
    # A raw lux reading well above 1 should read as bright, not error.
    out = s.update({"light": 800.0, "t": 0})
    assert out["light"] == "bright"


def test_reaction_rate_limited():
    s = EmbodiedSenses(min_repeat_s=100)

    def shake_burst(times):
        fired = []
        for t in times:
            out = s.update({"motion": GRAVITY + 25, "t": t})
            if "shaken" in out["events"]:
                fired.append(t)
        return fired

    s.update({"motion": GRAVITY, "t": 0})
    assert shake_burst(range(1, 6))          # first shake fires
    s.update({"motion": GRAVITY, "t": 6})    # back to rest
    assert not shake_burst(range(7, 12))     # second shake within cooldown is suppressed
