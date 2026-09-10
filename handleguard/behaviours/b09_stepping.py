from __future__ import annotations

from handleguard.behaviours.base import (
    BehaviourDetector,
    FrameContext,
    confidence_from,
    foot_overlap,
    SustainedCondition,
)


class SteppingDetector(BehaviourDetector):
    """B09 — a person's foot region rests on a product.

    No pose model: this is bounding-box contact geometry, so it reports
    *spatial contact sustained over time*, never intent. Requiring duration is
    what separates standing on a carton from walking past one.
    """

    id = "B09"
    name = "stepping"
    config_key = "stepping"
    requires_roles = {"product", "actor"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self._held = SustainedCondition()

    def reset(self) -> None:
        self._held.reset()

    def update(self, ctx: FrameContext):
        events = []
        min_overlap = float(self.cfg["min_foot_overlap"])
        min_duration = float(self.cfg["min_duration_seconds"])

        persons = ctx.persons()
        if not persons:
            return events

        for product_id in ctx.products():
            for person_id in persons:
                overlap = foot_overlap(ctx, person_id, product_id)
                if overlap < min_overlap:
                    continue

                span = self._held.observe((person_id, product_id), ctx.t)
                if span < min_duration:
                    continue

                margin = min(overlap / min_overlap - 1.0, span / min_duration - 1.0)
                severity = float(self.cfg["base_severity"]) + min(
                    max(overlap - min_overlap, 0.0), 1.0
                ) * 0.15
                events.append(
                    self.event(
                        ctx,
                        (person_id, product_id),
                        start_t=ctx.t - span,
                        severity=severity,
                        confidence=confidence_from(margin, ctx.tracks[product_id].conf),
                        evidence={
                            "foot_overlap": round(overlap, 3),
                            "sustained_seconds": round(span, 3),
                            "basis": "bounding-box contact geometry, no pose estimation",
                        },
                        zone=ctx.f(product_id).zone,
                    )
                )
        return events
