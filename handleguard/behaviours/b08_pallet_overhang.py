from __future__ import annotations

from handleguard.behaviours.base import (
    BehaviourDetector,
    FrameContext,
    confidence_from,
    supported_fraction,
    SustainedCondition,
)


class PalletOverhangDetector(BehaviourDetector):
    """B08 — product footprint extends beyond the pallet supporting it.

    Distinct from B06: unstable stack is product-on-product, this is
    product-on-pallet, and the pallet is the thing that has to contain it.
    """

    id = "B08"
    name = "pallet_overhang"
    config_key = "pallet_overhang"
    requires_roles = {"product", "support"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self._held = SustainedCondition()

    def reset(self) -> None:
        self._held.reset()

    def update(self, ctx: FrameContext):
        events = []
        min_support = float(self.cfg["min_support_fraction"])
        min_duration = float(self.cfg["min_duration_seconds"])

        pallets = ctx.by_role("support")
        if not pallets:
            return events

        for product_id in ctx.products():
            fractions = [(supported_fraction(ctx, product_id, p), p) for p in pallets]
            fraction, pallet_id = max(fractions)
            # Untouched by any pallet: not an overhang, it is simply elsewhere.
            if fraction <= 0.0 or fraction >= min_support:
                continue

            span = self._held.observe((product_id, pallet_id), ctx.t)
            if span < min_duration:
                continue

            deficit = (min_support - fraction) / max(min_support, 1e-6)
            margin = min(deficit, span / min_duration - 1.0)
            severity = float(self.cfg["base_severity"]) + min(max(deficit, 0.0), 1.0) * 0.15
            events.append(
                self.event(
                    ctx,
                    (product_id, pallet_id),
                    start_t=ctx.t - span,
                    severity=severity,
                    confidence=confidence_from(margin, ctx.tracks[product_id].conf),
                    evidence={
                        "support_fraction": round(fraction, 3),
                        "required_support_fraction": min_support,
                        "overhang_fraction": round(1.0 - fraction, 3),
                        "sustained_seconds": round(span, 3),
                    },
                    zone=ctx.f(product_id).zone,
                )
            )
        return events
