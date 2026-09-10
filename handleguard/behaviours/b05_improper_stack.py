from __future__ import annotations

from handleguard.behaviours.base import (
    BehaviourDetector,
    FrameContext,
    confidence_from,
    stack_pair,
    SustainedCondition,
)


class ImproperStackDetector(BehaviourDetector):
    """B05 — a larger item stacked on top of a smaller one.

    The real rule is about *weight* order, which video cannot observe. We use
    bounding-box area as a size proxy, and the evidence dict says so explicitly
    so the explanation and the UI never imply we measured mass.
    """

    id = "B05"
    name = "improper_stack"
    config_key = "improper_stack"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._held = SustainedCondition()

    def reset(self) -> None:
        self._held.reset()

    def update(self, ctx: FrameContext):
        events = []
        min_area_ratio = float(self.cfg["min_area_ratio"])
        min_overlap = float(self.cfg["min_overlap_x"])
        min_duration = float(self.cfg["min_duration_seconds"])

        for upper_id in ctx.products():
            for lower_id in ctx.below(upper_id):
                if ctx.tracks[lower_id].role != "product":
                    continue
                area_ratio, overlap_x, _ = stack_pair(ctx, upper_id, lower_id)
                if area_ratio < min_area_ratio or overlap_x < min_overlap:
                    continue

                span = self._held.observe((upper_id, lower_id), ctx.t)
                if span < min_duration:
                    continue

                margin = min(area_ratio / min_area_ratio - 1.0, span / min_duration - 1.0)
                severity = float(self.cfg["base_severity"]) + min(
                    max(area_ratio - min_area_ratio, 0.0), 1.0
                ) * 0.15
                events.append(
                    self.event(
                        ctx,
                        (upper_id, lower_id),
                        start_t=ctx.t - span,
                        severity=severity,
                        confidence=confidence_from(margin, ctx.tracks[upper_id].conf),
                        evidence={
                            "area_ratio_size_proxy": round(area_ratio, 3),
                            "horizontal_overlap": round(overlap_x, 3),
                            "sustained_seconds": round(span, 3),
                            "basis": "size proxy from bounding boxes, not measured weight",
                        },
                        zone=ctx.f(upper_id).zone,
                    )
                )
        return events
