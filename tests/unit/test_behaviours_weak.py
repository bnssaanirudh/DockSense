"""B04/B10/B11 — the three behaviours labelled *lightly validated*.

These are proxy heuristics with known confounds. The tests below pin the
suppression logic that stops them being noisy, because that is the part most
likely to break the demo:

- B04 must not echo a drop that B01 already reported.
- B10 must go quiet the moment handling equipment is visible.
- B11 must only fire on an ordered pair for the *same* entity.
"""

from __future__ import annotations

from dataclasses import replace

import handleguard.config as config
from handleguard.behaviours.b04_rough_handling import RoughHandlingDetector
from handleguard.behaviours.b10_manual_heavy_handling import ManualHeavyHandlingDetector
from handleguard.behaviours.b11_unsafe_sequence import UnsafeSequenceDetector
from handleguard.types import BehaviourEvent
from tests.fixtures import synth
from tests.fixtures.synth import feat, make_multi_ctx, synth_track


def _det(cls):
    return cls(dict(config.behaviours()[cls.config_key]))


def _event(bid, name, start, end, tracks, conf=0.8):
    return BehaviourEvent(
        behaviour_id=bid,
        name=name,
        track_ids=tuple(tracks),
        start_frame=0,
        end_frame=1,
        start_t=start,
        end_t=end,
        severity=0.6,
        confidence=conf,
        evidence={},
        zone=None,
    )


# --- B04 rough handling ----------------------------------------------------

def _jolt_ctx(recent=()):
    values = [
        feat(t=0.0, cx=0.5, cy=0.30, vy=0.2, ay=0.2),
        feat(t=0.5, cx=0.5, cy=0.36, vy=5.0, ay=6.0),
        feat(t=1.0, cx=0.5, cy=0.40, vy=0.1, ay=6.0),
    ]
    ctx = synth.make_ctx(values)
    return ctx if not recent else type(ctx)(**{**ctx.__dict__, "recent_events": recent})


def test_b04_silent_on_smooth_motion():
    assert _det(RoughHandlingDetector).update(synth.scenario_gentle_place()) == []


def test_b04_fires_on_spike_plus_hard_stop():
    events = _det(RoughHandlingDetector).update(_jolt_ctx())
    assert len(events) == 1 and events[0].behaviour_id == "B04"


def test_b04_defers_to_drop_already_reported():
    """It overlaps B01 by construction; it must not duplicate the alert."""
    recent = (_event("B01", "drop", 0.0, 1.0, [1]),)
    assert _det(RoughHandlingDetector).update(_jolt_ctx(recent)) == []


# --- B10 large item without equipment --------------------------------------

def _handling_ctx(with_equipment: bool):
    person = synth_track(track_id=1, cls="person", role="actor", xyxy=(400, 200, 520, 560))
    box = synth_track(track_id=2, xyxy=(420, 300, 700, 540))
    tracks = [person, box]
    per = {
        1: [feat(track_id=1, t=i * 0.5, cx=0.4, cy=0.5, h_px=360) for i in range(8)],
        2: [feat(track_id=2, t=i * 0.5, cx=0.45, cy=0.55, h_px=240) for i in range(8)],
    }
    if with_equipment:
        trolley = synth_track(track_id=3, cls="trolley", role="equipment", xyxy=(800, 400, 950, 560))
        tracks.append(trolley)
        per[3] = [feat(track_id=3, t=i * 0.5, cx=0.7, cy=0.6, h_px=160) for i in range(8)]
    return make_multi_ctx(tracks, per)


def test_b10_silent_when_equipment_is_present():
    assert _det(ManualHeavyHandlingDetector).update(_handling_ctx(True)) == []


def test_b10_fires_when_no_equipment_in_frame():
    detector = _det(ManualHeavyHandlingDetector)
    ctx = _handling_ctx(False)
    # B10 also requires the handling to persist, not just to exist for a frame.
    events = []
    t = ctx.t
    while t <= ctx.t + 3.0:
        events = detector.update(replace(ctx, t=t))
        t += 0.1
    assert len(events) == 1
    ev = events[0]
    assert ev.behaviour_id == "B10"
    assert ev.evidence["equipment_in_frame"] is False
    # Must not imply weight was measured.
    assert "not observable" in ev.evidence["basis"]


# --- B11 unsafe sequence ---------------------------------------------------

def _seq_ctx(recent):
    ctx = synth.scenario_drop()
    return type(ctx)(**{**ctx.__dict__, "t": 12.0, "recent_events": tuple(recent)})


def test_b11_silent_when_events_belong_to_different_entities():
    recent = [_event("B03", "drag", 0.0, 1.0, [1]), _event("B01", "drop", 2.0, 3.0, [99])]
    assert _det(UnsafeSequenceDetector).update(_seq_ctx(recent)) == []


def test_b11_silent_on_an_unlisted_pair():
    recent = [_event("B09", "stepping", 0.0, 1.0, [1]), _event("B12", "surface", 2.0, 3.0, [1])]
    assert _det(UnsafeSequenceDetector).update(_seq_ctx(recent)) == []


def test_b11_fires_on_drag_then_drop_for_one_entity():
    recent = [_event("B03", "drag", 4.0, 5.0, [1]), _event("B01", "drop", 6.0, 7.0, [1])]
    events = _det(UnsafeSequenceDetector).update(_seq_ctx(recent))
    assert len(events) == 1
    assert events[0].evidence["description"] == "dragged, then dropped"
