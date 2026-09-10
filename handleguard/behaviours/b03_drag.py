from __future__ import annotations

from handleguard.behaviours.base import BehaviourDetector, FrameContext, confidence_from, travel_heights


class DragDetector(BehaviourDetector):
    id = "B03"
    name = "drag"
    config_key = "drag"

    def update(self, ctx: FrameContext):
        events = []
        duration = float(self.cfg["min_duration_seconds"])
        for track_id in ctx.products():
            feat = ctx.f(track_id)
            window = ctx.window(track_id, duration)
            if len(window) < 2:
                continue
            dx, _, distance = travel_heights(window, ctx.fw, ctx.fh)
            vertical_variation = max(f.cy for f in window) - min(f.cy for f in window)
            vertical_heights = vertical_variation * ctx.fh / max(feat.h_px, 1e-6)
            floor_gaps = sorted(f.floor_gap for f in window if f.floor_gap is not None)
            # Median, not max. The floor reference comes from the nearest person's
            # feet, and person detection flickers; requiring EVERY frame in the
            # window to be near the floor makes the whole test hostage to the one
            # frame where the reference person was missed.
            median_gap = floor_gaps[len(floor_gaps) // 2] if floor_gaps else None
            near_floor = median_gap is not None and median_gap <= float(self.cfg["floor_proximity"])
            margin = min(
                abs(dx) / float(self.cfg["min_distance_heights"]),
                float(self.cfg["max_vertical_variation"]) / max(vertical_heights, 1e-6),
            )
            if (
                distance >= float(self.cfg["min_distance_heights"])
                and vertical_heights <= float(self.cfg["max_vertical_variation"])
                and near_floor
            ):
                severity = float(self.cfg["base_severity"]) + min(distance / 4.0, 1.0) * 0.15
                events.append(
                    self.event(
                        ctx,
                        (track_id,),
                        start_t=window[0].t,
                        severity=severity,
                        confidence=confidence_from(margin, ctx.tracks[track_id].conf),
                        evidence={
                            "horizontal_travel_heights": round(abs(dx), 3),
                            "vertical_variation_heights": round(vertical_heights, 3),
                            "median_floor_gap": round(median_gap, 3) if median_gap is not None else None,
                        },
                        zone=feat.zone,
                    )
                )
        return events
