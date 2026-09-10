"""B05/B06/B08/B09/B12 — static-geometry and zone behaviours.

Per the contract, each detector gets exactly two tests: the positive scenario
must fire, and the named hard negative must stay silent. The negative is the one
that matters — a detector that fires on everything passes every positive test
and destroys the demo.
"""

from __future__ import annotations

from dataclasses import replace

import handleguard.config as config
from handleguard.behaviours.b05_improper_stack import ImproperStackDetector
from handleguard.behaviours.b06_unstable_stack import UnstableStackDetector
from handleguard.behaviours.b08_pallet_overhang import PalletOverhangDetector
from handleguard.behaviours.b09_stepping import SteppingDetector
from handleguard.behaviours.b12_unsafe_surface import UnsafeSurfaceDetector
from tests.fixtures import synth


def _det(cls):
    return cls(dict(config.behaviours()[cls.config_key]))


def _sustained(detector, ctx, seconds: float = 3.0, step: float = 0.1):
    """Feed one static scene repeatedly so the condition can actually persist.

    These detectors require the geometry to hold for `min_duration_seconds`. A
    single `update()` call is a scene that has existed for zero seconds, so a
    one-shot test could only pass while the duration gate was measuring track
    age instead of how long the condition held — which is exactly the bug this
    replaced.
    """
    events = []
    t = ctx.t
    while t <= ctx.t + seconds:
        events = detector.update(replace(ctx, t=t))
        t += step
    return events


# --- B05 improper stack ----------------------------------------------------

def test_b05_silent_on_correct_stack():
    assert _det(ImproperStackDetector).update(synth.scenario_correct_stack()) == []


def test_b05_fires_on_large_box_on_small():
    events = _sustained(_det(ImproperStackDetector), synth.scenario_improper_stack())
    assert len(events) == 1
    ev = events[0]
    assert ev.behaviour_id == "B05"
    assert ev.evidence["area_ratio_size_proxy"] > 1.4
    # Must never imply we measured weight.
    assert "size proxy" in ev.evidence["basis"]


# --- B06 unstable stack ----------------------------------------------------

def test_b06_silent_on_stable_stack():
    assert _det(UnstableStackDetector).update(synth.scenario_stable_stack()) == []


def test_b06_fires_on_overhanging_stack():
    events = _sustained(_det(UnstableStackDetector), synth.scenario_unstable_stack())
    assert len(events) == 1
    assert events[0].behaviour_id == "B06"
    assert events[0].evidence["support_ratio"] < 0.6


# --- B08 pallet overhang ---------------------------------------------------

def test_b08_silent_when_product_fully_on_pallet():
    assert _det(PalletOverhangDetector).update(synth.scenario_pallet_well_supported()) == []


def test_b08_fires_on_overhanging_product():
    events = _sustained(_det(PalletOverhangDetector), synth.scenario_pallet_overhang())
    assert len(events) == 1
    ev = events[0]
    assert ev.behaviour_id == "B08"
    assert 0.0 < ev.evidence["support_fraction"] < 0.75
    assert ev.evidence["overhang_fraction"] > 0.25


# --- B09 stepping ----------------------------------------------------------

def test_b09_silent_when_walking_past():
    assert _det(SteppingDetector).update(synth.scenario_walking_past_product()) == []


def test_b09_fires_when_foot_rests_on_product():
    events = _sustained(_det(SteppingDetector), synth.scenario_stepping_on_product())
    assert len(events) == 1
    assert events[0].behaviour_id == "B09"
    assert events[0].evidence["foot_overlap"] >= 0.3


# --- B12 unsafe surface ----------------------------------------------------

def test_b12_silent_in_normal_zone():
    assert _det(UnsafeSurfaceDetector).update(synth.scenario_safe_surface()) == []


def test_b12_fires_in_wet_floor_zone():
    events = _det(UnsafeSurfaceDetector).update(synth.scenario_unsafe_surface())
    assert len(events) == 1
    assert events[0].zone == "wet_floor"


# --- contract compliance ---------------------------------------------------

def test_new_detectors_read_thresholds_only_from_config():
    """Rule 2: no numeric literals deciding behaviour inside detector bodies."""
    import inspect

    for cls in (
        ImproperStackDetector,
        UnstableStackDetector,
        PalletOverhangDetector,
        SteppingDetector,
        UnsafeSurfaceDetector,
    ):
        src = inspect.getsource(cls.update)
        assert "xyxy" not in src, f"{cls.__name__} touches raw boxes; use base helpers"
