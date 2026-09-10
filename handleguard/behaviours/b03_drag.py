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
            floor_gaps = [f.floor_gap for f in window if f.floor_gap is not None]
            near_floor = bool(floor_gaps) and max(floor_gaps) <= float(self.cfg["floor_proximity"])
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
                            "max_floor_gap": round(max(floor_gaps), 3) if floor_gaps else None,
                        },
                        zone=feat.zone,
                    )
                )
        return events
