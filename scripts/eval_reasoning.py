#!/usr/bin/env python3
"""Measure the reasoning layer on physics-rendered clips with exact perception.

Why this exists, and what it does NOT claim
-------------------------------------------
The system has two halves: perception (YOLO-World) and reasoning (tracking,
temporal features, twelve behaviour detectors, dedup, risk). They fail for
different reasons and should be measured separately.

This harness measures **reasoning only**. It feeds the pipeline the exact box
geometry the renderer drew, so every number here is unaffected by detection
quality. That is deliberate: YOLO-World correctly returns nothing on flat
rectangles, so running it over these clips would measure the renderer's
photorealism, not the reasoning.

    Perception is validated separately, on real industrial CCTV.
    Reasoning is validated here, on physics-rendered clips.
    Neither substitutes for real footage of the behaviours themselves.

Ground truth is frame-exact: the renderer knows which frame the box was released
and which frame it landed. Thresholds in configs/behaviours.yaml were never tuned
against these clips, so this is not a fit-to-the-test measurement.

The set is 4 positives and 5 hard negatives. The negatives carry most of the
value — gentle placement must not read as a drop, carrying must not read as
dragging, a correct stack must not read as an improper one, and a static scene
must produce nothing at all.

    python scripts/eval_reasoning.py
    python scripts/eval_reasoning.py --ablate      # same clips, mechanisms off
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from handleguard.evaluation.metrics import EventLabel, evaluate_events, labels_from_csv
from handleguard.pipeline import PipelineFlags, run
from handleguard.types import Detection, Frame

SYNTH = ROOT / "data" / "synthetic"


class ExactDetector:
    """Replays the renderer's own geometry. Perception error is zero by design."""

    def __init__(self, boxes_json: Path):
        payload = json.loads(boxes_json.read_text())
        self.src_w = float(payload["w"])
        self.src_h = float(payload["h"])
        # Keyed by time so frame skipping in the reader cannot desynchronise us.
        self.frames = [(f["t"], f["boxes"]) for f in payload["frames"]]

    def __call__(self, frame: Frame) -> list[Detection]:
        if not self.frames:
            return []
        _, boxes = min(self.frames, key=lambda p: abs(p[0] - frame.t))
        sx, sy = frame.w / self.src_w, frame.h / self.src_h
        out = []
        for role, x0, y0, x1, y1 in boxes:
            cls = "person" if role == "actor" else "carton"
            out.append(
                Detection(
                    cls=cls,
                    role=role,
                    conf=0.95,
                    xyxy=(x0 * sx, y0 * sy, x1 * sx, y1 * sy),
                )
            )
        return out


def predictions_for(video: Path, flags: PipelineFlags) -> list[EventLabel]:
    boxes = SYNTH / f"{video.stem}_boxes.json"
    if not boxes.is_file():
        raise SystemExit(f"missing {boxes} — run scripts/render_synthetic.py first")
    incidents = run(
        video,
        detector=ExactDetector(boxes),
        flags=flags,
        write_clips=False,
        video_id=video.name,
        session="synthetic-reasoning",
        # No zone polygons: the synthetic scene has no designated areas, and the
        # placeholder demo_cam_1 polygons happen to overlay it, so leaving them on
        # produced phantom zone_violation incidents for any box that sat still.
        camera="synthetic",
    )
    return [
        EventLabel(video=video.name, behaviour=inc.name, start_t=inc.start_t, end_t=inc.end_t)
        for inc in incidents
    ]


def run_variant(videos: list[Path], flags: PipelineFlags, gt, iou: float) -> dict:
    preds: list[EventLabel] = []
    per_clip: dict[str, int] = {}
    for v in videos:
        p = predictions_for(v, flags)
        per_clip[v.name] = len(p)
        preds.extend(p)
    report = evaluate_events(gt, preds, iou_threshold=iou)
    return {"label": flags.label(), "per_clip": per_clip, **report.to_dict()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iou", type=float, default=0.3, help="temporal IoU for a match")
    ap.add_argument("--ablate", action="store_true", help="also run mechanism-off variants")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "evaluation")
    args = ap.parse_args()

    videos = sorted(SYNTH.glob("*.mp4"))
    if not videos:
        raise SystemExit("no synthetic clips — run scripts/render_synthetic.py")
    gt = labels_from_csv(SYNTH / "ground_truth.csv")

    variants = {"baseline": PipelineFlags()}
    if args.ablate:
        variants.update(
            no_tracking=PipelineFlags(use_tracking=False),
            no_smoothing=PipelineFlags(use_smoothing=False),
            no_event_graph=PipelineFlags(use_event_graph=False),
        )

    results = {}
    for name, flags in variants.items():
        print(f"\n=== {name} ===", flush=True)
        res = run_variant(videos, flags, gt, args.iou)
        results[name] = res
        m = res["micro"]
        print(f"  precision {m['precision']:.3f}  recall {m['recall']:.3f}  f1 {m['f1']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "synthetic physics clips, exact perception injected",
        "measures": "reasoning layer only — tracking, temporal features, behaviours, dedup",
        "does_not_measure": "detection quality; that is validated separately on real CCTV",
        "caveat": "Not a substitute for real footage of these behaviours.",
        "iou_threshold": args.iou,
        "clips": {"positives": 4, "hard_negatives": 5},
        "variants": results,
    }
    (args.out / "reasoning_eval.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out / 'reasoning_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
