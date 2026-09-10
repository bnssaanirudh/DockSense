from __future__ import annotations

from handleguard.behaviours.base import (
    BehaviourDetector,
    FrameContext,
    confidence_from,
    stack_pair,
    SustainedCondition,
)


class UnstableStackDetector(BehaviourDetector):
    """B06 — a stacked item is insufficiently supported by the tier below it.

    Fires on low support ratio (the upper box hangs off the lower one) sustained
    long enough to be a real stack rather than a box mid-transit.
    """

    id = "B06"
    name = "unstable_stack"
    config_key = "unstable_stack"

    # A large box on a small one necessarily has a poor support ratio, so this
    # detector fires on every improper stack as well. Both are true, but they
    # describe one physical configuration, and B05 names the cause while this
    # names a symptom of it. Report the cause once, as with B04 deferring to
    # B01/B02.
    _SUPERSEDED_BY = {"B05"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self._held = SustainedCondition()

    def reset(self) -> None:
        self._held.reset()

    def update(self, ctx: FrameContext):
        events = []
        claimed = {
            tuple(sorted(ev.track_ids))
            for ev in ctx.recent_events
            if ev.behaviour_id in self._SUPERSEDED_BY
        }
        max_support = float(self.cfg["max_support_ratio"])
        min_duration = float(self.cfg["min_duration_seconds"])

        for upper_id in ctx.products():
            # Low overlap threshold on purpose: a badly overhanging box is the
            # case this detector exists for, and the default would hide it.
            candidates = ctx.below(upper_id, min_overlap=0.05)
            supports = [i for i in candidates if ctx.tracks[i].role in {"product", "support"}]
            if not supports:
                continue
            # Best-supported neighbour decides stability: a box resting across two
            # others is stable if either holds it well.
            ratios = [(stack_pair(ctx, upper_id, i)[2], i) for i in supports]
            support_ratio, lower_id = max(ratios)
            if support_ratio >= max_support:
                continue
            if tuple(sorted((upper_id, lower_id))) in claimed:
                continue

            span = self._held.observe((upper_id, lower_id), ctx.t)
            if span < min_duration:
                continue

            deficit = (max_support - support_ratio) / max(max_support, 1e-6)
            margin = min(deficit, span / min_duration - 1.0)
            severity = float(self.cfg["base_severity"]) + min(max(deficit, 0.0), 1.0) * 0.15
            events.append(
                self.event(
                    ctx,
                    (upper_id, lower_id),
                    start_t=ctx.t - span,
                    severity=severity,
                    confidence=confidence_from(margin, ctx.tracks[upper_id].conf),
                    evidence={
                        "support_ratio": round(support_ratio, 3),
                        "required_support_ratio": max_support,
                        "sustained_seconds": round(span, 3),
                    },
                    zone=ctx.f(upper_id).zone,
                )
            )
        return events
