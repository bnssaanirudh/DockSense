#!/usr/bin/env python3
"""Render physically-correct synthetic handling clips with exact ground truth.

Why this exists
---------------
No public dataset labels product drop / throw / drag / stacking events in a
warehouse (verified 7 Sep 2026). Rather than tune thresholds against footage we
do not have, we render sequences whose physics we control exactly.

This is NOT AI-generated video. Every trajectory here is integrated Newtonian
motion:

    v += g * dt
    y += v * dt

with a real gravitational constant and a real pixels-per-metre scale. That
matters because the drop detector measures vertical velocity and impact
deceleration — calibrating those against a generative model's guess at what
falling looks like would produce thresholds that transfer to nothing.

What it buys us: ground truth accurate to the frame. We know precisely which
frame the box was released and which frame it hit, so precision/recall computed
against these clips are real numbers, not estimates.

What it does NOT buy us: proof that the system works on real warehouse video.
Rendered clips validate detector LOGIC. Real footage validates the whole thing.
The submission must state which claims rest on which — see the scope ledger in
plan.md.

Usage
-----
    python scripts/render_synthetic.py                 # render all scenarios
    python scripts/render_synthetic.py --only drop     # one scenario
    python scripts/render_synthetic.py --preview       # dump first frame as PNG
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Scale and physics constants
# --------------------------------------------------------------------------- #

W, H = 1280, 720
FPS = 30

#: A standard shipping carton is roughly 0.4 m tall. We render it 80 px tall,
#: which fixes the scale of the whole scene.
PX_PER_M = 80.0 / 0.4  # = 200 px per metre
G = 9.81 * PX_PER_M  # px / s^2

FLOOR_Y = 620.0  # y of the floor line in pixels

RNG = np.random.default_rng(20260907)  # deterministic renders


# --------------------------------------------------------------------------- #
# Scene objects
# --------------------------------------------------------------------------- #


@dataclass
class Box:
    """An axis-aligned carton with Newtonian motion."""

    x: float  # left edge, px
    y: float  # TOP edge, px (y grows downward)
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    falling: bool = False
    resting_on: float | None = None  # y of the surface it rests on
    tone: int = 0  # texture variant
    label: str = "carton"

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def step(self, dt: float, floor: float) -> bool:
        """Advance one tick. Returns True if an impact happened this tick."""
        if not self.falling:
            return False
        self.vy += G * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        surface = self.resting_on if self.resting_on is not None else floor
        if self.bottom >= surface:
            self.y = surface - self.h
            # Inelastic: cartons do not bounce meaningfully. A small bounce
            # would be more realistic but adds a second impact the ground truth
            # would have to account for, so we keep it simple and honest.
            self.vy = 0.0
            self.vx = 0.0
            self.falling = False
            return True
        return False


@dataclass
class Scenario:
    name: str
    behaviour: str | None  # None => hard negative, must produce no event
    duration: float
    note: str = ""
    events: list[tuple[float, float]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _cardboard(w: int, h: int, tone: int) -> np.ndarray:
    """Procedural corrugated-cardboard texture.

    A flat brown rectangle reads as 'rectangle' to an open-vocabulary detector.
    Grain, seam lines and a tape strip push it towards 'cardboard box'. Whether
    that is enough is an empirical question — scripts/check_synthetic_detect.py
    answers it rather than assuming.
    """
    base = np.array([(150, 176, 199), (120, 148, 174), (139, 165, 190)][tone % 3], np.float32)
    img = np.tile(base, (h, w, 1))

    # Corrugation: faint vertical banding
    bands = (np.sin(np.arange(w) * 0.55) * 4.0).astype(np.float32)
    img += bands[None, :, None]

    # Paper grain
    img += RNG.normal(0, 5.0, (h, w, 1)).astype(np.float32)

    # Flap seam down the middle, and a tape strip over it
    cv2.line(img, (w // 2, 0), (w // 2, h), (110, 135, 158), 1)
    tape_w = max(int(w * 0.12), 3)
    x0 = w // 2 - tape_w // 2
    img[:, x0 : x0 + tape_w] = img[:, x0 : x0 + tape_w] * 0.55 + np.float32([185, 195, 205]) * 0.45

    # Edge shading so the box reads as a solid, not a sticker
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (95, 118, 140), 2)
    return np.clip(img, 0, 255).astype(np.uint8)


def _background() -> np.ndarray:
    """Warehouse-ish backdrop: concrete floor, wall, and a horizon line."""
    bg = np.zeros((H, W, 3), np.uint8)
    bg[: int(FLOOR_Y)] = (188, 186, 182)  # wall
    bg[int(FLOOR_Y) :] = (150, 152, 156)  # floor

    # Concrete speckle on the floor
    floor_h = H - int(FLOOR_Y)
    noise = RNG.normal(0, 7, (floor_h, W, 1)).astype(np.float32)
    bg[int(FLOOR_Y) :] = np.clip(bg[int(FLOOR_Y) :] + noise, 0, 255).astype(np.uint8)

    cv2.line(bg, (0, int(FLOOR_Y)), (W, int(FLOOR_Y)), (120, 122, 126), 3)

    # Faint floor markings, gives the tracker some static structure
    for gx in range(0, W, 220):
        cv2.line(bg, (gx, int(FLOOR_Y)), (gx - 60, H), (140, 142, 146), 1)
    return bg


# Exact boxes drawn for the current frame, as (role, x0, y0, x1, y1).
# Emitted alongside the video so evaluation can isolate the reasoning layer from
# perception: YOLO-World correctly returns nothing on flat rectangles, so any
# detector run over these clips would measure the renderer, not the reasoning.
_FRAME_BOXES: list[tuple[str, float, float, float, float]] = []


def _draw(frame: np.ndarray, box: Box) -> None:
    x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
    if w > 0 and h > 0:
        _FRAME_BOXES.append(("product", float(x), float(y), float(x + w), float(y + h)))
    if w <= 0 or h <= 0:
        return
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    tex = _cardboard(w, h, box.tone)
    frame[y0:y1, x0:x1] = tex[y0 - y : y1 - y, x0 - x : x1 - x]

    # Contact shadow — helps the eye and the detector separate box from floor
    if abs(box.bottom - FLOOR_Y) < 3:
        cv2.ellipse(
            frame,
            (int(box.x + w / 2), int(FLOOR_Y) + 3),
            (int(w * 0.5), 5),
            0, 0, 360, (120, 122, 126), -1,
        )


def _draw_person(frame: np.ndarray, cx: float, top: float, height: float = 300.0) -> None:
    """A crude standing figure. Present so person-product association has
    something to bind to; not intended to be photoreal."""
    c = (70, 70, 78)
    hh = height
    head_r = int(hh * 0.09)
    _FRAME_BOXES.append(("actor", float(cx - hh * 0.13), float(top), float(cx + hh * 0.13), float(top + hh)))
    cv2.circle(frame, (int(cx), int(top + head_r)), head_r, c, -1)
    cv2.rectangle(
        frame,
        (int(cx - hh * 0.11), int(top + head_r * 2)),
        (int(cx + hh * 0.11), int(top + hh * 0.62)),
        c, -1,
    )
    for dx in (-0.06, 0.06):
        cv2.rectangle(
            frame,
            (int(cx + hh * dx - hh * 0.035), int(top + hh * 0.62)),
            (int(cx + hh * dx + hh * 0.035), int(top + hh)),
            c, -1,
        )


# --------------------------------------------------------------------------- #
# Scenarios — each returns (scenario, per-frame simulation callback)
# --------------------------------------------------------------------------- #


def _carton(x, bottom_y, w=70, h=80, tone=0) -> Box:
    return Box(x=x, y=bottom_y - h, w=w, h=h, tone=tone)


def scenario_drop(dt: float):
    """Box released at chest height, falls freely, impacts floor."""
    sc = Scenario("drop", "drop", 4.0, "release at ~1.4 m, free fall to floor")
    box = _carton(600, FLOOR_Y - 280)  # held above the floor
    release_t, impact_t = 1.0, None

    def step(t: float, frame: np.ndarray):
        nonlocal impact_t
        if t >= release_t and not box.falling and box.bottom < FLOOR_Y - 1:
            box.falling = True
        hit = box.step(dt, FLOOR_Y)
        if hit and impact_t is None:
            impact_t = t
            sc.events.append((release_t, t))
        _draw_person(frame, 520, FLOOR_Y - 320)
        _draw(frame, box)

    return sc, step


def scenario_gentle_place(dt: float):
    """HARD NEGATIVE. Same start height, lowered under control. Must not fire."""
    sc = Scenario("gentle_place", None, 4.0, "controlled lowering, ~0.25 m/s")
    box = _carton(600, FLOOR_Y - 280)
    speed = 0.25 * PX_PER_M  # px/s, an order of magnitude below free fall

    def step(t: float, frame: np.ndarray):
        if t >= 1.0 and box.bottom < FLOOR_Y:
            box.y = min(box.y + speed * dt, FLOOR_Y - box.h)
        _draw_person(frame, 520, FLOOR_Y - 320)
        _draw(frame, box)

    return sc, step


def scenario_throw(dt: float):
    """Box launched horizontally, follows a parabola, lands away from thrower."""
    sc = Scenario("throw", "throw", 4.0, "launched at ~3 m/s horizontal")
    box = _carton(360, FLOOR_Y - 240)
    release_t = 1.0

    def step(t: float, frame: np.ndarray):
        if t >= release_t and not box.falling and box.bottom < FLOOR_Y - 1:
            box.falling = True
            box.vx = 3.0 * PX_PER_M
            box.vy = -0.8 * PX_PER_M  # slight upward toss
        hit = box.step(dt, FLOOR_Y)
        if hit:
            sc.events.append((release_t, t))
        _draw_person(frame, 300, FLOOR_Y - 320)
        _draw(frame, box)

    return sc, step


def scenario_drag(dt: float):
    """Box slid along the floor, never lifted."""
    sc = Scenario("drag", "drag", 5.0, "1.5 m of floor-contact translation")
    box = _carton(260, FLOOR_Y)
    t0, t1 = 1.0, 4.0
    speed = 0.5 * PX_PER_M

    def step(t: float, frame: np.ndarray):
        if t0 <= t <= t1:
            box.x += speed * dt
            box.y = FLOOR_Y - box.h  # stays pinned to the floor
        _draw_person(frame, box.x - 70, FLOOR_Y - 320)
        _draw(frame, box)

    sc.events.append((t0, t1))
    return sc, step


def scenario_carry(dt: float):
    """HARD NEGATIVE for drag. Same horizontal travel, but lifted clear."""
    sc = Scenario("carry", None, 5.0, "same translation, carried at knee height")
    box = _carton(260, FLOOR_Y - 180)
    speed = 0.5 * PX_PER_M

    def step(t: float, frame: np.ndarray):
        if 1.0 <= t <= 4.0:
            box.x += speed * dt
        _draw_person(frame, box.x - 70, FLOOR_Y - 320)
        _draw(frame, box)

    return sc, step


# Placement descent: 0.4 m/s = 80 px/s starting at t=1.0, so the box must start
# 80 px clear of its target to land at t=2.0 — the moment the ground-truth label
# says the stack exists. It started 300 px clear, which takes 3.75 s: the stack
# was only real for the last 0.25 s of a 5 s clip while the label claimed 3 s of
# it. The label was right about the intent and the physics disagreed, so the
# physics is what moved.
PLACEMENT_RISE = 80.0  # px above the target surface at t=0 (y grows downward)


def scenario_improper_stack(dt: float):
    """Large carton placed on top of a small one."""
    sc = Scenario("improper_stack", "improper_stack", 5.0, "large on small")
    small = _carton(600, FLOOR_Y, w=60, h=60, tone=1)
    large = Box(x=560, y=(small.y - 110) - PLACEMENT_RISE, w=140, h=110, tone=2)

    def step(t: float, frame: np.ndarray):
        target = small.y - large.h
        if t >= 1.0 and large.y < target:
            large.y = min(large.y + 0.4 * PX_PER_M * dt, target)
        _draw(frame, small)
        _draw(frame, large)

    sc.events.append((2.0, 5.0))
    return sc, step


def scenario_good_stack(dt: float):
    """HARD NEGATIVE. Small on large, well centred. Must stay silent."""
    sc = Scenario("good_stack", None, 5.0, "small on large, correct order")
    large = _carton(560, FLOOR_Y, w=140, h=110, tone=2)
    small = Box(x=600, y=(large.y - 60) - PLACEMENT_RISE, w=60, h=60, tone=1)

    def step(t: float, frame: np.ndarray):
        target = large.y - small.h
        if t >= 1.0 and small.y < target:
            small.y = min(small.y + 0.4 * PX_PER_M * dt, target)
        _draw(frame, large)
        _draw(frame, small)

    return sc, step


def scenario_unstable_stack(dt: float):
    """Upper carton overhangs its support by well over half its width."""
    sc = Scenario("unstable_stack", "unstable_stack", 5.0, "support ratio ~0.3")
    lower = _carton(560, FLOOR_Y, w=140, h=110, tone=2)
    upper = Box(x=560 + 100, y=(lower.y - 80) - PLACEMENT_RISE, w=130, h=80, tone=0)

    def step(t: float, frame: np.ndarray):
        target = lower.y - upper.h
        if t >= 1.0 and upper.y < target:
            upper.y = min(upper.y + 0.4 * PX_PER_M * dt, target)
        _draw(frame, lower)
        _draw(frame, upper)

    sc.events.append((2.0, 5.0))
    return sc, step


def scenario_static(dt: float):
    """HARD NEGATIVE. Nothing moves. Must produce zero incidents.

    The single most valuable clip in this set — 'here is normal operation and
    the system stayed silent' answers the question every judge is privately
    asking."""
    sc = Scenario("static", None, 6.0, "nothing moves for 6 s")
    boxes = [
        _carton(300, FLOOR_Y, tone=0),
        _carton(420, FLOOR_Y, w=140, h=110, tone=2),
        _carton(450, FLOOR_Y - 110, w=60, h=60, tone=1),
    ]

    def step(t: float, frame: np.ndarray):
        for b in boxes:
            _draw(frame, b)

    return sc, step


SCENARIOS = {
    "drop": scenario_drop,
    "gentle_place": scenario_gentle_place,
    "throw": scenario_throw,
    "drag": scenario_drag,
    "carry": scenario_carry,
    "improper_stack": scenario_improper_stack,
    "good_stack": scenario_good_stack,
    "unstable_stack": scenario_unstable_stack,
    "static": scenario_static,
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def render(name: str, out_dir: Path, preview: bool = False) -> Scenario:
    dt = 1.0 / FPS
    sc, step = SCENARIOS[name](dt)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.mp4"

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    n = int(sc.duration * FPS)
    first = None
    per_frame: list[dict] = []
    for i in range(n):
        t = i * dt
        frame = _background()
        _FRAME_BOXES.clear()
        step(t, frame)
        per_frame.append({"t": round(t, 4), "boxes": list(_FRAME_BOXES)})
        if first is None:
            first = frame.copy()
        writer.write(frame)
    writer.release()

    # Exact geometry, not estimated — this is what makes these clips usable as a
    # reasoning-layer benchmark rather than only a visual aid.
    (out_dir / f"{name}_boxes.json").write_text(
        json.dumps({"video": f"{name}.mp4", "fps": FPS, "w": W, "h": H, "frames": per_frame})
        + "\n"
    )

    if preview and first is not None:
        cv2.imwrite(str(out_dir / f"{name}_frame0.png"), first)

    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="render just this scenario")
    ap.add_argument("--preview", action="store_true", help="also dump frame 0 as PNG")
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out
    names = [args.only] if args.only else list(SCENARIOS)

    rows = []
    for name in names:
        sc = render(name, out_dir, preview=args.preview)
        kind = sc.behaviour or "NEGATIVE"
        if sc.events:
            for t0, t1 in sc.events:
                rows.append([f"{name}.mp4", sc.behaviour, f"{t0:.3f}", f"{t1:.3f}", sc.note])
            spans = ", ".join(f"{a:.2f}-{b:.2f}s" for a, b in sc.events)
            print(f"  {name:16s} {kind:16s} {spans}")
        else:
            rows.append([f"{name}.mp4", "", "", "", sc.note])
            print(f"  {name:16s} {kind:16s} (no event expected)")

    gt = out_dir / "ground_truth.csv"
    write_header = not gt.exists() or args.only is None
    mode = "w" if write_header else "a"
    with open(gt, mode, newline="") as fh:
        wr = csv.writer(fh)
        if write_header:
            wr.writerow(["video", "behaviour", "t_start", "t_end", "note"])
        wr.writerows(rows)

    print(f"\nwrote {len(names)} clips + ground_truth.csv to {out_dir}")
    print("Ground truth is exact: release and impact frames are known, not estimated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
