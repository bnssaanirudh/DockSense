#!/usr/bin/env python3
"""Crop recorded session clips down to the CCTV pane.

The session-1 footage is a phone recording of an NVMS playback window, not
direct camera output: each frame carries software chrome and a burned-in caption
card around the actual camera view. Running perception on that measures how well
YOLO-World finds cartons in a screenshot of a user interface.

This crops each clip to its camera pane using configs/rois.yaml and writes clean
clips to data/processed/<session>/. Ground-truth times are preserved, because
cropping is spatial only — no frames are dropped and the frame rate is unchanged.

    python scripts/prepare_footage.py --session session1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def crop_clip(src: Path, dst: Path, box: dict) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x0, y0, x1, y1 = int(box["x0"]), int(box["y0"]), int(box["x1"]), int(box["y1"])
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        raise SystemExit(f"empty ROI for {src.name}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame[y0:y1, x0:x1])
        n += 1
    cap.release()
    writer.release()
    return w, h, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session1")
    ap.add_argument("--rois", type=Path, default=ROOT / "configs" / "rois.yaml")
    args = ap.parse_args()

    rois = yaml.safe_load(args.rois.read_text())["clips"]
    src_dir = ROOT / "data" / "raw" / args.session
    dst_dir = ROOT / "data" / "processed" / args.session

    missing = sorted({p.stem for p in src_dir.glob("*.mp4")} - set(rois))
    if missing:
        raise SystemExit(f"no ROI in {args.rois.name} for: {', '.join(missing)}")

    for src in sorted(src_dir.glob("*.mp4")):
        w, h, n = crop_clip(src, dst_dir / src.name, rois[src.stem])
        print(f"  {src.stem:26s} -> {w}x{h}  {n} frames")

    print(f"\ncropped clips in {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
