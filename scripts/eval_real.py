#!/usr/bin/env python3
"""Clip-level behaviour check on real recorded footage (session 1).

Why clip level and not temporal IoU
-----------------------------------
The synthetic evaluation matches predicted spans against frame-exact ground
truth. That is impossible here: these takes were not logged at record time, so
`takes.csv` carries the behaviour the recorder says each clip contains and no
start/end times. Inventing spans by scrubbing the footage afterwards would mean
scoring the system against labels the same person drew after seeing what the
system did, which is worse than admitting the labels are coarse.

So this asks the coarser question honestly: **for each clip, did the system
report the behaviour the recorder said was there?**

What this therefore cannot measure
----------------------------------
* **Precision.** Session 1 contains no hard negatives — no clip of normal
  handling that must stay silent. Every clip is a positive, so a detector that
  fired constantly would score perfectly here. The per-minute alarm rate below
  is reported instead, as the closest available signal, and it is not precision.
* **Timing.** A behaviour reported at the wrong moment counts as a hit.

Split: 3 clips tuned on, 4 held out and scored once. Named in `takes.csv`.

    python scripts/eval_real.py --run      # run the pipeline (slow) and cache
    python scripts/eval_real.py            # score from the cache
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "artifacts" / "evaluation" / "real_footage_incidents.json"


def expected(takes: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(takes, newline="") as fh:
        for row in csv.DictReader(fh):
            stem = Path(row["filename"]).stem
            entry = out.setdefault(stem, {"split": row["session"], "behaviours": []})
            entry["behaviours"].append(row["behaviour"])
    return out


def run_all(clips: Path, db: Path | None = None) -> dict:
    from handleguard.db import IncidentStore
    from handleguard.pipeline import run

    store = None
    if db is not None:
        db.parent.mkdir(parents=True, exist_ok=True)
        if db.exists():
            db.unlink()  # a demo DB of real incidents, rebuilt not appended
        store = IncidentStore(db)
    cache: dict[str, list] = {}
    for video in sorted(clips.glob("*.mp4")):
        incidents = run(
            video,
            write_clips=True,
            video_id=video.stem,
            session="session1",
            camera="session1",
            store=store,
        )
        cache[video.stem] = [
            {
                "behaviour_id": inc.behaviour_id,
                "name": inc.name,
                "start_t": round(inc.start_t, 2),
                "end_t": round(inc.end_t, 2),
                "risk": inc.risk.score,
                "band": inc.risk.band,
                "confidence": inc.risk.confidence,
            }
            for inc in incidents
        ]
        print(f"  {video.stem:26s} {len(cache[video.stem]):3d} incidents", flush=True)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run the pipeline and refresh the cache")
    ap.add_argument("--clips", type=Path, default=ROOT / "data" / "processed" / "session1")
    ap.add_argument("--takes", type=Path, default=ROOT / "data" / "raw" / "session1" / "takes.csv")
    ap.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "processed" / "incidents.db",
        help="also write incidents here, so the console shows real events",
    )
    args = ap.parse_args()

    if args.run:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(run_all(args.clips, args.db), indent=2) + "\n")
    if not CACHE.is_file():
        raise SystemExit(f"no cache at {CACHE} — run with --run first")

    cache = json.loads(CACHE.read_text())
    want = expected(args.takes)
    durations = _durations(args.clips)

    per_split: dict[str, dict] = defaultdict(lambda: {"hit": 0, "miss": 0, "clips": 0})
    rows = []
    for stem, entry in sorted(want.items()):
        fired = {inc["name"] for inc in cache.get(stem, [])}
        hits = [b for b in entry["behaviours"] if b in fired]
        misses = [b for b in entry["behaviours"] if b not in fired]
        split = entry["split"]
        per_split[split]["hit"] += len(hits)
        per_split[split]["miss"] += len(misses)
        per_split[split]["clips"] += 1
        n = len(cache.get(stem, []))
        mins = durations.get(stem, 0.0) / 60.0
        rows.append(
            {
                "clip": stem,
                "split": split,
                "expected": entry["behaviours"],
                "fired": sorted(fired),
                "hits": hits,
                "misses": misses,
                "incidents": n,
                "alarms_per_minute": round(n / mins, 1) if mins else None,
            }
        )

    for r in rows:
        mark = "HIT " if not r["misses"] else "MISS"
        print(
            f"{mark} {r['clip']:26s} {r['split']:12s} "
            f"want={','.join(r['expected']):28s} fired={','.join(r['fired']) or '(none)':28s} "
            f"n={r['incidents']:3d} ({r['alarms_per_minute']}/min)"
        )

    summary = {}
    for split, counts in per_split.items():
        total = counts["hit"] + counts["miss"]
        summary[split] = {
            **counts,
            "recall": round(counts["hit"] / total, 3) if total else None,
        }
        print(f"\n{split}: {counts['hit']}/{total} expected behaviours reported")

    payload = {
        "dataset": "session 1, real warehouse CCTV re-recorded off a playback screen",
        "measures": "clip-level behaviour presence only",
        "does_not_measure": (
            "precision (no hard-negative clips were recorded) and timing "
            "(takes.csv carries no start/end times)"
        ),
        "clips": rows,
        "summary": summary,
    }
    out = ROOT / "artifacts" / "evaluation" / "real_footage.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


def _durations(clips: Path) -> dict[str, float]:
    import cv2

    out = {}
    for video in clips.glob("*.mp4"):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out[video.stem] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
