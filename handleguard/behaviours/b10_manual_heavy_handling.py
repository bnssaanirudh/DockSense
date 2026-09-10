from __future__ import annotations

from handleguard.behaviours.base import (
    BehaviourDetector,
    FrameContext,
    confidence_from,
    SustainedCondition,
)

EQUIPMENT_ROLES = {"equipment"}


class ManualHeavyHandlingDetector(BehaviourDetector):
    """B10 — a large item is handled with no handling equipment in frame.

    **Named for what it can actually see.** The underlying safety rule is about
    *weight*, and video cannot observe weight. This detector sees a large
    bounding box, a person in contact with it, and no trolley or pallet jack
    nearby. So it reports exactly that — "large item handled without equipment
    present" — rather than "improper manual handling", which would be a judgment
    about a person the system is not entitled to make.

    Size is a stand-in for mass and will be wrong for anything not uniformly
    dense. The evidence dict says so. Reported as *lightly validated*.
    """

    id = "B10"
    name = "manual_heavy_handling"
    config_key = "manual_heavy_handling"
    requires_roles = {"product", "actor"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self._held = SustainedCondition()

    def reset(self) -> None:
        self._held.reset()

    def update(self, ctx: FrameContext):
        events = []
        max_persons = int(self.cfg["max_persons"])
        min_duration = float(self.cfg["min_duration_seconds"])
        percentile = float(self.cfg["large_object_area_percentile"])

        products = ctx.products()
        if not products or not ctx.persons():
            return events

        # "Large" is relative to what else is in this scene — an absolute pixel
        # area would mean something different at every camera distance.
        areas = sorted(ctx.tracks[t].width * ctx.tracks[t].height for t in products)
        if not areas:
            return events
        cutoff = areas[min(int(len(areas) * percentile), len(areas) - 1)]

        equipment_present = any(
            ctx.tracks[t].role in EQUIPMENT_ROLES for t in ctx.tracks
        )

        for track_id in products:
            track = ctx.tracks[track_id]
            if track.width * track.height < cutoff:
                continue

            handlers = ctx.overlapping(track_id, role="actor")
            if not handlers or len(handlers) > max_persons:
                continue
            if equipment_present:
                continue

            span = self._held.observe(track_id, ctx.t)
            if span < min_duration:
                continue

            margin = span / min_duration - 1.0
            events.append(
                self.event(
                    ctx,
                    (handlers[0], track_id),
                    start_t=ctx.t - span,
                    severity=float(self.cfg["base_severity"]),
                    confidence=confidence_from(margin, track.conf),
                    evidence={
                        "handler_count": len(handlers),
                        "equipment_in_frame": False,
                        "sustained_seconds": round(span, 3),
                        "basis": "bounding-box size proxy; weight is not observable from video",
                    },
                    zone=ctx.f(track_id).zone,
                )
            )
        return events
