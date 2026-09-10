"""Regressions for three bugs that 140 passing tests could not see.

Each of these was invisible to the existing suite because that suite feeds a
single frame and only ever checks relative quantities. They were found by an
end-to-end reasoning evaluation, and they stay found because of this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from handleguard.behaviours.base import SustainedCondition, travel_heights
from handleguard.tracking.tracker import _ByteTrackResults
from handleguard.types import Detection, TrackFeatures


def _feat(t: float, cx: float, cy: float, h_px: float = 80.0) -> TrackFeatures:
    return TrackFeatures(track_id=1, t=t, h_px=h_px, cx=cx, cy=cy)


def test_bytetrack_boxes_are_centre_based():
    """BYTETracker feeds .xywh into xywh2ltwh, so it must be centre-based.

    Returning top-left here shifted every tracked box up and left by half its own
    size — which relative measures cancel out, so nothing caught it.
    """
    det = Detection(cls="carton", role="product", conf=0.9, xyxy=(100.0, 200.0, 180.0, 300.0))
    xywh = _ByteTrackResults.from_detections([det], []).xywh
    assert np.allclose(xywh[0], [140.0, 250.0, 80.0, 100.0])


def test_travel_heights_uses_width_for_x_and_height_for_y():
    """cx is normalised by frame width, cy by frame height. Each needs its axis.

    Using height for both shrank horizontal travel by the aspect ratio (0.5625 on
    16:9), so a drag that really moved 2 object-heights measured 1.125.
    """
    window = [_feat(0.0, 0.25, 0.5), _feat(1.0, 0.375, 0.5)]
    dx, dy, _ = travel_heights(window, 1280, 720)
    assert dx == 2.0          # 0.125 * 1280 px = 160 px = 2 x an 80 px box
    assert dy == 0.0

    # Same displacement vertically must NOT give the same answer on a 16:9 frame.
    vertical = [_feat(0.0, 0.5, 0.25), _feat(1.0, 0.5, 0.375)]
    _, dy2, _ = travel_heights(vertical, 1280, 720)
    assert dy2 == 1.125


def test_sustained_condition_clocks_the_condition_not_the_track():
    held = SustainedCondition(max_gap=0.5)
    span = held.observe("pair", 0.0)
    assert span == 0.0                           # a scene that just appeared is 0 s old
    for i in range(1, 21):                       # observed every 0.1 s, as a detector is
        span = held.observe("pair", i * 0.1)
    assert span == pytest.approx(2.0)

    # A lapse longer than max_gap restarts the clock: a flicker must not
    # accumulate into a sustained observation.
    assert held.observe("pair", 3.0) == 0.0
    assert held.observe("pair", 3.1) == pytest.approx(0.1)
