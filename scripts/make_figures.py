#!/usr/bin/env python3
"""Render detector-overlay figures from real footage into artifacts/screenshots/.

These exist so a reader can check the perception claim without running anything:
each figure is one real frame from session 1 with every detection drawn on it,
class and confidence included. Products are outlined in one colour and people in
another, because the role split is what the behaviour layer actually consumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from handleguard.perception.detector import YoloWorldDetector
from handleguard.types import Frame

PRODUCT = (86, 196, 122)   # BGR
ACTOR = (219, 152, 66)
OTHER = (150, 150, 150)


def annotate(frame, detections):
    for det in detections:
        x0, y0, x1, y1 = (int(v) for v in det.xyxy)
        colour = {"product": PRODUCT, "actor": ACTOR}.get(det.role, OTHER)
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
        label = f"{det.cls} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x0, max(y0 - th - 6, 0)), (x0 + tw + 6, y0), colour, -1)
        cv2.putText(frame, label, (x0 + 3, max(y0 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    clips = ROOT / "data" / "processed" / "session1"
    out = ROOT / "artifacts" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    detector = YoloWorldDetector()

    for video in sorted(clips.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        dets = detector(Frame(index=0, t=0.0, image=frame, w=frame.shape[1], h=frame.shape[0]))
        path = out / f"detections_{video.stem}.jpg"
        cv2.imwrite(str(path), annotate(frame, dets), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {path.name:46s} {len(dets)} detections")

    print(f"\nwrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
