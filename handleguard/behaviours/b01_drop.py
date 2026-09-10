from __future__ import annotations

from handleguard.behaviours.base import (
    BehaviourDetector,
    FrameContext,
    confidence_from,
    motion_start_t,
    travel_heights,
)


class DropDetector(BehaviourDetector):
    id = "B01"
    name = "drop"
    config_key = "drop"

    def update(self, ctx: FrameContext):
        events = []
        lookback = float(self.cfg["cooldown_seconds"])
        for track_id in ctx.products():
            feat = ctx.f(track_id)
            window = ctx.window(track_id, lookback)
            dx, dy, _ = travel_heights(window, ctx.fh)
            prev = window[-2] if len(window) >= 2 else feat
            decel = max(prev.vy - feat.vy, 0.0)
            margin = min(
                dy / float(self.cfg["min_fall_heights"]),
                feat.vy / float(self.cfg["min_downward_velocity"]),
                decel / float(self.cfg["impact_decel"]) if decel else 0.0,
            )
            # Horizontal share of travel across the whole window, not the
            # instantaneous value. A throw's ratio dips near the top of its arc,
            # so sampling one frame reads it as a drop; the flight as a whole
            # does not.
            travel = abs(dx) + abs(dy)
            horizontal_share = abs(dx) / travel if travel > 1e-6 else 0.0
            mostly_vertical = horizontal_share <= float(self.cfg["max_horizontal_ratio"])

            if mostly_vertical and dy >= float(self.cfg["min_fall_heights"]) and (
                feat.vy >= float(self.cfg["min_downward_velocity"])
                or decel >= float(self.cfg["impact_decel"])
            ):
                severity = float(self.cfg["base_severity"]) + min(max(dy - 1.0, 0.0), 1.0) * 0.15
                events.append(
                    self.event(
                        ctx,
                        (track_id,),
                        start_t=motion_start_t(window, min_speed=float(self.cfg["min_downward_velocity"]) * 0.25),
                        severity=severity,
                        confidence=confidence_from(margin, ctx.tracks[track_id].conf),
                        evidence={
                            "fall_heights": round(dy, 3),
                            "downward_velocity": round(feat.vy, 3),
                            "impact_decel": round(decel, 3),
                            "horizontal_share": round(horizontal_share, 3),
                        },
                        zone=feat.zone,
                    )
                )
        return events
