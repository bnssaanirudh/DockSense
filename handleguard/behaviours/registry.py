"""Behaviour detector registry."""

from __future__ import annotations

from handleguard.behaviours.base import BehaviourDetector
from handleguard.behaviours.b01_drop import DropDetector
from handleguard.behaviours.b02_throw import ThrowDetector
from handleguard.behaviours.b03_drag import DragDetector
from handleguard.behaviours.b04_rough_handling import RoughHandlingDetector
from handleguard.behaviours.b05_improper_stack import ImproperStackDetector
from handleguard.behaviours.b06_unstable_stack import UnstableStackDetector
from handleguard.behaviours.b07_zone_violation import ZoneViolationDetector
from handleguard.behaviours.b08_pallet_overhang import PalletOverhangDetector
from handleguard.behaviours.b09_stepping import SteppingDetector
from handleguard.behaviours.b10_manual_heavy_handling import ManualHeavyHandlingDetector
from handleguard.behaviours.b11_unsafe_sequence import UnsafeSequenceDetector
from handleguard.behaviours.b12_unsafe_surface import UnsafeSurfaceDetector

# Order matters: a detector that supersedes another must run BEFORE it, so the
# more specific reading is already on the frame's event list when the general
# one decides whether to stay quiet. ThrowDetector before DropDetector (a thrown
# box also falls); ImproperStackDetector before UnstableStackDetector (a large
# box on a small one is necessarily poorly supported).
DETECTOR_CLASSES: tuple[type[BehaviourDetector], ...] = (
    ThrowDetector,
    DropDetector,
    DragDetector,
    RoughHandlingDetector,
    ImproperStackDetector,
    UnstableStackDetector,
    ZoneViolationDetector,
    PalletOverhangDetector,
    SteppingDetector,
    ManualHeavyHandlingDetector,
    UnsafeSequenceDetector,
    UnsafeSurfaceDetector,
)


def build_all(cfg: dict) -> list[BehaviourDetector]:
    detectors: list[BehaviourDetector] = []
    for cls in DETECTOR_CLASSES:
        section = dict(cfg.get(cls.config_key, {}))
        detector = cls(section)
        if detector.enabled:
            detectors.append(detector)
    return detectors
