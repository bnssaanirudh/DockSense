"""Track-to-feature conversion for behaviour detectors.

This is the normalization boundary. Everything below here may speak pixels;
everything above here receives object-height-scaled motion features.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from handleguard import config
from handleguard.perception import geometry as g
from handleguard.types import Frame, Track, TrackFeatures


@dataclass
class FeatureExtractor:
    """Compute normalized temporal and context features for active tracks."""

    camera: str = "demo_cam_1"
    window_frames: int | None = None
    min_track_age_frames: int | None = None
    floor_y_norm: float = 0.86
    zones_cfg: dict[str, Any] | None = None
    _prev: dict[int, TrackFeatures] = field(default_factory=dict)
    _windows: dict[int, deque[TrackFeatures]] = field(default_factory=lambda: defaultdict(deque))

    def __post_init__(self) -> None:
        smoothing = config.behaviours().get("smoothing", {})
        if self.window_frames is None:
            self.window_frames = int(smoothing.get("window_frames", 5))
        if self.min_track_age_frames is None:
            self.min_track_age_frames = int(smoothing.get("min_track_age_frames", 4))
        if self.zones_cfg is None:
            self.zones_cfg = config.zones()

    def reset(self) -> None:
        self._prev.clear()
        self._windows.clear()

    def update(self, tracks: list[Track], frame: Frame) -> dict[int, TrackFeatures]:
        out: dict[int, TrackFeatures] = {}
        by_id = {track.id: track for track in tracks}
        for track in tracks:
            feat = self._compute_one(track, frame, by_id)
            self._remember(feat)
            # min_track_age_frames was declared in behaviours.yaml and never read
            # by anything — a documented knob that did nothing. It matters: a
            # track one frame old has no velocity history, so its first real
            # measurement is a full-magnitude jump, and on soft real footage that
            # jump is detector jitter rather than motion. Held-back features are
            # still remembered, so history is intact the moment the track is old
            # enough to be trusted.
            if track.age >= self.min_track_age_frames:
                out[track.id] = feat

        active_ids = set(by_id)
        for track_id in list(self._prev):
            if track_id not in active_ids:
                del self._prev[track_id]
        for track_id in list(self._windows):
            if track_id not in active_ids:
                del self._windows[track_id]
        return out

    def _floor_gap(
        self, track: Track, frame: Frame, tracks: dict[int, Track], y2: float, h_px: float
    ) -> float:
        """Height above the local ground, in object-heights.

        A single horizontal floor line only describes an overhead-ish view. In a
        perspective view the ground is metres away at the top of frame and under
        the camera at the bottom, so one line puts a box on the floor at one
        depth and a metre in the air at another — which is why drag never fired
        on real footage.

        The nearest person's feet are a calibration-free local ground reference:
        whoever is standing beside the box is standing on the same floor, at the
        same depth. Falls back to the configured line when nobody is in frame.
        """
        actors = [t for t in tracks.values() if t.role == "actor" and t.id != track.id]
        if actors:
            cx, _ = g.center(track.xyxy)
            nearest = min(actors, key=lambda t: abs(g.center(t.xyxy)[0] - cx))
            return max(nearest.xyxy[3] - y2, 0.0) / h_px
        return max((self.floor_y_norm * frame.h) - y2, 0.0) / h_px

    def _compute_one(self, track: Track, frame: Frame, tracks: dict[int, Track]) -> TrackFeatures:
        x1, y1, x2, y2 = track.xyxy
        h_px = max(y2 - y1, 1e-6)
        cx_px, cy_px = g.center(track.xyxy)
        cx = _clamp01(cx_px / max(frame.w, 1))
        cy = _clamp01(cy_px / max(frame.h, 1))
        area_px = g.area(track.xyxy)

        prev = self._prev.get(track.id)
        if prev is None or frame.t <= prev.t:
            vx = vy = ax = ay = 0.0
        else:
            dt = frame.t - prev.t
            norm_h = max((h_px + prev.h_px) / 2.0, 1e-6)
            vx = (cx - prev.cx) * frame.w / norm_h / dt
            vy = (cy - prev.cy) * frame.h / norm_h / dt
            ax = (vx - prev.vx) / dt
            ay = (vy - prev.vy) / dt

        zone = self._zone_for(track, frame)
        floor_gap = self._floor_gap(track, frame, tracks, y2, h_px)
        supported_by = tuple(
            other.id
            for other in tracks.values()
            if other.id != track.id and other.role != "actor" and g.is_above(track.xyxy, other.xyxy, max_gap=max(h_px * 0.15, 2.0))
        )
        held_by_person = any(
            self._looks_held(track, other)
            for other in tracks.values()
            if other.id != track.id and other.role == "actor"
        )

        provisional = TrackFeatures(
            track_id=track.id,
            t=frame.t,
            h_px=h_px,
            cx=cx,
            cy=cy,
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            horizontal_ratio=0.0,
            floor_gap=floor_gap,
            area_px=area_px,
            zone=zone,
            supported_by=supported_by,
            held_by_person=held_by_person,
        )
        return TrackFeatures(
            **{
                **provisional.__dict__,
                "horizontal_ratio": self._horizontal_ratio(track.id, provisional, frame),
            }
        )

    def _remember(self, feat: TrackFeatures) -> None:
        self._prev[feat.track_id] = feat
        q = self._windows[feat.track_id]
        q.append(feat)
        while len(q) > int(self.window_frames or 1):
            q.popleft()

    def _horizontal_ratio(self, track_id: int, current: TrackFeatures, frame: Frame) -> float:
        values = list(self._windows.get(track_id, ())) + [current]
        if len(values) < 2:
            return 0.0
        first, last = values[0], values[-1]
        dx = abs(last.cx - first.cx) * frame.w
        dy = abs(last.cy - first.cy) * frame.h
        total = dx + dy
        return dx / total if total > 0 else 0.0

    def _zone_for(self, track: Track, frame: Frame) -> str | None:
        cameras = (self.zones_cfg or {}).get("cameras", {})
        zones = cameras.get(self.camera, {}).get("zones", [])
        bx, by = g.bottom_center(track.xyxy)
        pt = (bx / max(frame.w, 1), by / max(frame.h, 1))
        selected: str | None = None
        for zone in zones:
            poly = [(float(x), float(y)) for x, y in zone.get("polygon", [])]
            if g.point_in_polygon(pt, poly):
                selected = str(zone.get("name"))
                if str(zone.get("type", "")).startswith("restricted"):
                    return selected
        return selected

    @staticmethod
    def _looks_held(product: Track, actor: Track) -> bool:
        """Does this person plausibly have hold of this product?

        The old test was horizontal overlap plus *any* vertical intersection,
        which in a loading bay means "somebody is standing behind it at a
        different depth". Measured on the session-1 tune clips, a carton being
        thrown across a gap read as held in 100% of frames, so B02 could never
        fire on a real throw. Requiring a real share of the product's own area
        to be inside the person's box is a stricter and more literal reading of
        "holding it".
        """
        share = g.intersection_area(product.xyxy, actor.xyxy) / max(g.area(product.xyxy), 1e-6)
        return share >= 0.2


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
