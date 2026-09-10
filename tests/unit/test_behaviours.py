from __future__ import annotations

from handleguard import config
from handleguard.behaviours.b01_drop import DropDetector
from handleguard.behaviours.b02_throw import ThrowDetector
from handleguard.behaviours.b03_drag import DragDetector
from handleguard.behaviours.b07_zone_violation import ZoneViolationDetector
from handleguard.behaviours.registry import build_all
from tests.fixtures import synth


def _cfg(key: str) -> dict:
    return config.behaviours()[key]


def test_registry_builds_all_enabled_detectors():
    detectors = build_all(config.behaviours())
    assert len(detectors) == 12
    # Order is deliberate (a superseding detector runs first), so compare as a set.
    assert sorted(det.id for det in detectors) == [f"B{i:02d}" for i in range(1, 13)]


def test_drop_negative_gentle_place_stays_silent():
    assert DropDetector(_cfg("drop")).update(synth.scenario_gentle_place()) == []


def test_drop_positive_fires():
    events = DropDetector(_cfg("drop")).update(synth.scenario_drop())
    assert len(events) == 1
    assert events[0].behaviour_id == "B01"
    assert events[0].evidence["fall_heights"] >= _cfg("drop")["min_fall_heights"]


def test_throw_negative_carry_stays_silent():
    assert ThrowDetector(_cfg("throw")).update(synth.scenario_carry()) == []


def test_throw_positive_fires():
    events = ThrowDetector(_cfg("throw")).update(synth.scenario_throw())
    assert len(events) == 1
    assert events[0].behaviour_id == "B02"
    assert events[0].evidence["unsupported_frames"] >= _cfg("throw")["min_unsupported_frames"]


def test_drag_negative_carry_stays_silent():
    assert DragDetector(_cfg("drag")).update(synth.scenario_carry()) == []


def test_drag_positive_fires():
    events = DragDetector(_cfg("drag")).update(synth.scenario_drag())
    assert len(events) == 1
    assert events[0].behaviour_id == "B03"
    assert events[0].evidence["horizontal_travel_heights"] >= _cfg("drag")["min_distance_heights"]


def test_zone_transient_crossing_stays_silent():
    assert ZoneViolationDetector(_cfg("zone_violation")).update(synth.scenario_transient_zone_crossing()) == []


def test_zone_violation_positive_fires():
    events = ZoneViolationDetector(_cfg("zone_violation")).update(synth.scenario_zone_violation())
    assert len(events) == 1
    assert events[0].behaviour_id == "B07"
    assert events[0].zone == "walkway"
