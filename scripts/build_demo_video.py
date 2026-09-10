#!/usr/bin/env python3
"""Assemble the submission demo video from real project artefacts.

Nothing here is a mock-up. The footage is session-1 loading-bay video, the
detection overlays come from running the real detector, the evidence clip is one
the pipeline wrote (face blur included), and the console frame is a screenshot of
the running app showing real incidents. The caption cards are rendered from HTML
so they match the deck.

Requires ffmpeg and Chrome (headless, for the cards).

    python scripts/build_demo_video.py --out submission/GEGAI_Bay9/DockSense_Demo.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H, FPS = 1920, 1080, 30

CARD_CSS = """
*{box-sizing:border-box;margin:0}
html,body{width:1920px;height:1080px}
body{
  background:#0C1214; color:#E8EEEF; display:flex; flex-direction:column;
  justify-content:center; padding:120px 140px; gap:34px;
  font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;
}
.eyebrow{
  font-family:"SF Mono",Menlo,monospace; font-size:26px; letter-spacing:.22em;
  text-transform:uppercase; color:#E8B400; font-weight:600;
}
h1{font-size:104px; line-height:1.02; letter-spacing:-.025em; font-weight:800; max-width:24ch}
h2{font-size:74px; line-height:1.06; letter-spacing:-.02em; font-weight:800; max-width:26ch}
p{font-size:36px; line-height:1.42; color:#A3B6BC; max-width:44ch}
p.wide{max-width:60ch}
.rule{height:5px; background:#E8B400; width:190px}
.big{font-size:190px; font-weight:800; color:#E2685A; letter-spacing:-.04em; line-height:.9}
.row{display:flex; align-items:baseline; gap:52px}
table{border-collapse:collapse; font-size:34px; font-family:"SF Mono",Menlo,monospace}
td,th{padding:16px 40px 16px 0; text-align:left; border-bottom:1px solid #2A383D}
th{color:#6E858D; font-size:24px; letter-spacing:.1em; text-transform:uppercase; font-weight:500}
tr.key td{color:#E2685A; font-weight:700}
ul{margin:0; padding:0; list-style:none; display:flex; flex-direction:column; gap:22px}
li{font-size:33px; color:#A3B6BC; padding-left:26px; border-left:4px solid #2E7767; line-height:1.32}
li.open{border-left-color:#E2685A}
li b{color:#E8EEEF; font-weight:600}
"""

CARDS: list[tuple[str, float, str]] = [
    ("title", 6, """
      <div class="eyebrow">Team Bay 9 &nbsp;·&nbsp; GEGAI</div>
      <h1>The camera already saw it.</h1>
      <div class="rule"></div>
      <p class="wide">DockSense turns loading-bay video into reviewable incidents. A single frame cannot
      tell you whether a box was placed or dropped — that lives in a sequence.</p>
    """),
    ("problem", 11, """
      <div class="eyebrow">The gap</div>
      <h2>Recording is not the problem. Nobody watches it.</h2>
      <p class="wide">CCTV produces evidence after a damage claim — product broken, cause disputed, and the
      handling pattern that caused it repeating unnoticed for weeks.</p>
    """),
    ("perception", 9, """
      <div class="eyebrow">Step one &nbsp;·&nbsp; real loading-bay footage</div>
      <h2>Detect products and people. No model was trained.</h2>
      <p class="wide">Open-vocabulary prompts, running offline on an 8&nbsp;GB laptop.</p>
    """),
    ("reasoning", 11, """
      <div class="eyebrow">Step two</div>
      <h2>Track each entity, then reason over its short history.</h2>
      <p class="wide">Thresholds are in object-heights, not pixels — a box falling 1.5&times; its own height
      fell the same amount whether the camera is 3&nbsp;m or 8&nbsp;m away. Risk and confidence stay separate
      fields, because "how bad if real" and "how sure it is real" are different questions.</p>
    """),
    ("evidence", 9, """
      <div class="eyebrow">Step three &nbsp;·&nbsp; stored evidence</div>
      <h2>Every incident keeps its own clip.</h2>
      <p class="wide">Faces are blurred before the clip is written. Behaviour, never identity.</p>
    """),
    ("console", 9, """
      <div class="eyebrow">Step four</div>
      <h2>A supervisor reviews, not an alarm that fires.</h2>
      <p class="wide">Risk, confidence, evidence values and the SOP that applies — for real incidents from the
      footage you just watched.</p>
    """),
    ("ablation", 14, """
      <div class="eyebrow">Does the temporal reasoning earn its place?</div>
      <h2>Remove tracking and the system goes silent.</h2>
      <table>
        <tr><th>Variant</th><th>Tune n=9</th><th>Held out n=27</th></tr>
        <tr><td>baseline</td><td>1.000</td><td>0.933</td></tr>
        <tr class="key"><td>no_tracking</td><td>0.000</td><td>0.000</td></tr>
        <tr><td>no_smoothing</td><td>1.000</td><td>0.867</td></tr>
      </table>
      <p>Held-out F1. The tune column reads 1.000 because it was tuned on — which is why a held-out split exists.</p>
    """),
    ("honest", 17, """
      <div class="eyebrow">Then we pointed it at a real dock</div>
      <div class="row"><span class="big">1/9</span>
        <p>expected behaviours reported across seven real clips. We lead with this number.</p></div>
      <ul>
        <li><b>Fixed.</b> A config knob was declared and read by nothing.</li>
        <li><b>Fixed.</b> One horizontal floor line cannot describe a perspective view.</li>
        <li><b>Fixed.</b> "Held by a person" was true in 100% of frames.</li>
        <li class="open"><b>Not fixable by tuning.</b> Throw and drag are not separable at this resolution.</li>
      </ul>
    """),
    ("ask", 11, """
      <div class="eyebrow">What moves this next</div>
      <h2>Sixty seconds of ordinary handling.</h2>
      <p class="wide">One clip where nothing bad happens. Until it exists, precision on real footage is
      unmeasurable — every clip we have is a positive. Cheapest item on the list, highest value.</p>
      <div class="rule"></div>
      <p>Team Bay 9 &nbsp;·&nbsp; SPOC Anirudh Badampudi</p>
    """),
]


def sh(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render_card(name: str, body: str, work: Path) -> Path:
    html = work / f"{name}.html"
    html.write_text(f"<style>{CARD_CSS}</style>{body}")
    png = work / f"{name}.png"
    sh([CHROME, "--headless", "--disable-gpu", f"--window-size={W},{H}",
        f"--screenshot={png}", "--hide-scrollbars", "--virtual-time-budget=3000",
        f"file://{html}"])
    return png


def still_segment(png: Path, seconds: float, out: Path) -> None:
    sh(["ffmpeg", "-y", "-loop", "1", "-t", str(seconds), "-i", str(png),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0C1214,fps={FPS},format=yuv420p",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium", str(out)])


def caption_strip(text: str, work: Path, name: str) -> Path:
    """Render a caption bar as a PNG.

    ffmpeg's drawtext filter is absent from many builds — including the one on
    this machine — so captions are rendered in the browser like the cards and
    composited with `overlay`, which is always available.
    """
    html = work / f"cap_{name}.html"
    html.write_text(
        "<style>*{margin:0;box-sizing:border-box}"
        f"html,body{{width:{W}px;height:96px}}"
        "body{background:#0C1214;color:#E8EEEF;display:flex;align-items:center;"
        "padding:0 64px;font:500 38px/1 'Helvetica Neue',Helvetica,Arial,sans-serif;"
        "letter-spacing:.01em}</style>"
        f"<div>{text}</div>"
    )
    png = work / f"cap_{name}.png"
    sh([CHROME, "--headless", "--disable-gpu", f"--window-size={W},96",
        f"--screenshot={png}", "--hide-scrollbars", "--virtual-time-budget=2000",
        f"file://{html}"])
    return png


def video_segment(src: Path, out: Path, *, caption: str | None = None,
                  work: Path | None = None, name: str = "seg") -> None:
    base = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0C1214,fps={FPS}")
    if caption and work is not None:
        strip = caption_strip(caption, work, name)
        sh(["ffmpeg", "-y", "-i", str(src), "-i", str(strip),
            "-filter_complex", f"[0:v]{base}[v];[v][1:v]overlay=0:{H - 96}[o]",
            "-map", "[o]", "-an", "-c:v", "libx264", "-crf", "20",
            "-preset", "medium", "-pix_fmt", "yuv420p", str(out)])
        return
    sh(["ffmpeg", "-y", "-i", str(src),
        "-vf", f"{base},format=yuv420p",
        "-an", "-c:v", "libx264", "-crf", "20", "-preset", "medium", str(out)])


def annotate_clip(src: Path, out: Path) -> None:
    """Draw live detections over a real clip — the perception claim, visible."""
    from handleguard.perception.detector import YoloWorldDetector
    from handleguard.types import Frame

    detector = YoloWorldDetector()
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    index = 0
    dets: list = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Detect every third frame; boxes persist between, which keeps the render
        # honest about what the detector saw without tripling the runtime.
        if index % 3 == 0:
            dets = detector(Frame(index=index, t=index / fps, image=frame, w=w, h=h))
        for det in dets:
            x0, y0, x1, y1 = (int(v) for v in det.xyxy)
            colour = (86, 196, 122) if det.role == "product" else (219, 152, 66)
            cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
            cv2.putText(frame, f"{det.cls} {det.conf:.2f}", (x0 + 3, max(y0 - 7, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        writer.write(frame)
        index += 1
    cap.release()
    writer.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "submission" / "GEGAI_Bay9" / "DockSense_Demo.mp4")
    ap.add_argument("--console", type=Path, default=ROOT / "artifacts" / "screenshots" / "console_real_incidents.png")
    ap.add_argument("--evidence", type=Path, default=ROOT / "data" / "clips" / "INC-0A3C978C.mp4")
    args = ap.parse_args()

    clips = ROOT / "data" / "processed" / "session1"
    work = Path(tempfile.mkdtemp(prefix="docksense_video_"))
    cards = {name: render_card(name, body, work) for name, _, body in CARDS}
    durations = {name: secs for name, secs, _ in CARDS}

    print("annotating real footage...", flush=True)
    annotate_clip(clips / "roll_drop_carton.mp4", work / "ann_drop.mp4")
    annotate_clip(clips / "roll_drag_wet_floor.mp4", work / "ann_drag.mp4")

    order: list[Path] = []
    def add(seg: Path) -> None:
        order.append(seg)

    for name in ("title", "problem", "perception"):
        seg = work / f"seg_{name}.mp4"
        still_segment(cards[name], durations[name], seg)
        add(seg)

    for tag, src, caption in (
        ("drop", work / "ann_drop.mp4", "Session 1 &nbsp;|&nbsp; cartons and people detected, no training"),
        ("drag", work / "ann_drag.mp4", "Session 1 &nbsp;|&nbsp; wet floor, product moved along the ground"),
    ):
        seg = work / f"seg_{tag}.mp4"
        video_segment(src, seg, caption=caption, work=work, name=tag)
        add(seg)

    seg = work / "seg_reasoning.mp4"
    still_segment(cards["reasoning"], durations["reasoning"], seg)
    add(seg)

    seg = work / "seg_evidence_card.mp4"
    still_segment(cards["evidence"], durations["evidence"], seg)
    add(seg)
    if args.evidence.is_file():
        seg = work / "seg_evidence.mp4"
        video_segment(args.evidence, seg, work=work, name="evidence",
                      caption="Stored evidence clip &nbsp;|&nbsp; face blurred before writing")
        add(seg)

    seg = work / "seg_console_card.mp4"
    still_segment(cards["console"], durations["console"], seg)
    add(seg)
    if args.console.is_file():
        seg = work / "seg_console.mp4"
        still_segment(args.console, 9, seg)
        add(seg)

    for name in ("ablation", "honest", "ask"):
        seg = work / f"seg_{name}.mp4"
        still_segment(cards[name], durations[name], seg)
        add(seg)

    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in order))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:v", "libx264", "-crf", "21", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.out)])

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(args.out)],
        capture_output=True, text=True, check=True)
    seconds = float(probe.stdout.strip())
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {args.out}  ({seconds:.0f}s, limit 180s)")
    if seconds > 180:
        print("  WARNING: over the 3 minute submission limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
