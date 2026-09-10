from __future__ import annotations

from handleguard.behaviours.base import (
    BehaviourDetector,
    FrameContext,
    confidence_from,
    motion_start_t,
    travel_heights,
)


class ThrowDetector(BehaviourDetector):
    id = "B02"
    name = "throw"
    config_key = "throw"

    def update(self, ctx: FrameContext):
        events = []
        lookback = float(self.cfg["cooldown_seconds"])
        for track_id in ctx.products():
            feat = ctx.f(track_id)
            window = ctx.window(track_id, lookback)
            dx, _, distance = travel_heights(window, ctx.fw, ctx.fh)
            unsupported = [f for f in window if not f.held_by_person and (f.floor_gap is None or f.floor_gap > 0.1)]
            margin = min(
                feat.speed / float(self.cfg["min_speed"]),
                feat.horizontal_ratio / float(self.cfg["min_horizontal_ratio"]),
                len(unsupported) / float(self.cfg["min_unsupported_frames"]),
            )
            if (
                feat.speed >= float(self.cfg["min_speed"])
                and feat.horizontal_ratio >= float(self.cfg["min_horizontal_ratio"])
                and len(unsupported) >= int(self.cfg["min_unsupported_frames"])
            ):
                severity = float(self.cfg["base_severity"]) + min(abs(dx), 1.0) * 0.15
                events.append(
                    self.event(
                        ctx,
                        (track_id,),
                        start_t=motion_start_t(window, min_speed=float(self.cfg["min_speed"]) * 0.25),
                        severity=severity,
                        confidence=confidence_from(margin, ctx.tracks[track_id].conf),
                        evidence={
                            "speed": round(feat.speed, 3),
                            "horizontal_ratio": round(feat.horizontal_ratio, 3),
                            "unsupported_frames": len(unsupported),
                            "travel_heights": round(distance, 3),
                        },
                        zone=feat.zone,
                    )
                )
        return events
