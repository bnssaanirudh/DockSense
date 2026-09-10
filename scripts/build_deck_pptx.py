#!/usr/bin/env python3
"""Build the GEGAI submission deck as a real .pptx.

The web deck is the canonical version; this exists because a submission folder
needs something a reviewer can open and edit in PowerPoint. Content is kept
identical — if a number changes in one, change it in both.

    python scripts/build_deck_pptx.py --out submission/GEGAI_Bay9/DockSense_Concept_Deck.pptx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x12, 0x19, 0x1C)
INK2 = RGBColor(0x42, 0x55, 0x5C)
INK3 = RGBColor(0x72, 0x8A, 0x93)
HAZARD = RGBColor(0xC8, 0x96, 0x1A)
CRIT = RGBColor(0xB2, 0x3A, 0x2F)
GOOD = RGBColor(0x2E, 0x77, 0x67)
RULE = RGBColor(0xCB, 0xD6, 0xD9)
PAPER = RGBColor(0xFB, 0xFC, 0xFC)

DISPLAY = "Archivo"      # falls back to a system grotesque if absent
BODY = "IBM Plex Sans"
MONO = "IBM Plex Mono"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.72)


def _text(slide, left, top, width, height, runs, *, align=PP_ALIGN.LEFT, spacing=1.18):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, spec in enumerate(runs):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spec.get("line_spacing", spacing)
        if spec.get("space_before"):
            para.space_before = Pt(spec["space_before"])
        run = para.add_run()
        run.text = spec["text"]
        f = run.font
        f.name = spec.get("font", BODY)
        f.size = Pt(spec.get("size", 14))
        f.bold = spec.get("bold", False)
        f.color.rgb = spec.get("color", INK)
    return box


def _rule(slide, top, *, colour=RULE, height=Emu(9525), left=MARGIN, width=None):
    width = width or (W - MARGIN * 2)
    bar = slide.shapes.add_shape(1, left, top, width, height)  # 1 = rectangle
    bar.fill.solid()
    bar.fill.fore_color.rgb = colour
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def _slug(slide, tag: str, sub: str, index: str):
    _text(slide, MARGIN, Inches(0.44), Inches(9), Inches(0.3), [
        {"text": f"{tag}   {sub}", "font": MONO, "size": 10.5, "bold": True, "color": HAZARD},
    ])
    _text(slide, W - MARGIN - Inches(1.4), Inches(0.44), Inches(1.4), Inches(0.3), [
        {"text": index, "font": MONO, "size": 10.5, "color": INK3},
    ], align=PP_ALIGN.RIGHT)
    _rule(slide, Inches(0.78))


def _blank(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = PAPER
    return slide


def _table(slide, left, top, width, rows, col_widths, *, caption=None, key_row=None):
    if caption:
        _text(slide, left, top, width, Inches(0.25), [
            {"text": caption, "font": MONO, "size": 9.5, "color": INK3},
        ])
        top = top + Inches(0.34)
    shape = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width, Inches(0.34) * len(rows))
    table = shape.table
    for i, w in enumerate(col_widths):
        table.columns[i].width = w
    for r, row in enumerate(rows):
        table.rows[r].height = Inches(0.33)
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = value
            cell.margin_left = cell.margin_right = Inches(0.09)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PAPER
            para = cell.text_frame.paragraphs[0]
            run = para.runs[0] if para.runs else para.add_run()
            run.font.size = Pt(10.5 if r else 9.5)
            run.font.name = MONO if (c == 0 or c > 0 and r and _is_num(value)) else BODY
            run.font.bold = r == 0 or (key_row is not None and r == key_row and c == 0)
            run.font.color.rgb = INK3 if r == 0 else INK
            if key_row is not None and r == key_row and _is_num(value):
                run.font.color.rgb = CRIT
                run.font.bold = True
            if _is_num(value) and c:
                para.alignment = PP_ALIGN.RIGHT
    return shape


def _is_num(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def build(out: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---- 01 title ---------------------------------------------------------
    s = _blank(prs)
    _slug(s, "DOCKSENSE", "LOADING-BAY VIDEO INTELLIGENCE", "01 / 06")
    _text(s, MARGIN, Inches(1.35), Inches(8.2), Inches(2.2), [
        {"text": "The camera", "font": DISPLAY, "size": 62, "bold": True, "line_spacing": 0.95},
        {"text": "already saw it.", "font": DISPLAY, "size": 62, "bold": True, "line_spacing": 0.95},
    ])
    _rule(s, Inches(3.72), colour=HAZARD, height=Inches(0.62), left=MARGIN, width=Emu(45720))
    _text(s, MARGIN + Inches(0.18), Inches(3.72), Inches(6.4), Inches(0.8), [
        {"text": "A single frame cannot tell you whether a box was placed or dropped. "
                 "That lives in a sequence.", "font": DISPLAY, "size": 19, "bold": True, "line_spacing": 1.15},
    ])
    _text(s, MARGIN, Inches(4.75), Inches(11.4), Inches(1.4), [
        {"text": "Warehouses record everything and watch none of it. CCTV produces evidence after a damage "
                 "claim, by which point the product is broken, the cause is disputed, and the handling pattern "
                 "that caused it has repeated unnoticed for weeks. DockSense turns loading-bay video into "
                 "reviewable incidents: what happened, which entity, when, why it was judged risky, and how "
                 "sure the system is.", "size": 13, "color": INK2, "line_spacing": 1.4},
    ])
    _rule(s, Inches(6.28))
    facts = [("BEHAVIOURS", "12, tiered by evidence"), ("TRAINING", "None — text prompts"),
             ("RUNS ON", "Apple M3 / 8 GB, offline"), ("TESTS", "143 passing")]
    for i, (k, v) in enumerate(facts):
        x = MARGIN + Inches(3.05) * i
        _text(s, x, Inches(6.45), Inches(2.9), Inches(0.6), [
            {"text": k, "font": MONO, "size": 9, "color": INK3},
            {"text": v, "size": 12, "space_before": 3},
        ])

    # ---- 02 approach ------------------------------------------------------
    s = _blank(prs)
    _slug(s, "APPROACH", "SEQUENCES, NOT FRAMES", "02 / 06")
    _text(s, MARGIN, Inches(1.15), Inches(9.5), Inches(0.9), [
        {"text": "Track the entity, then reason over its short history.",
         "font": DISPLAY, "size": 33, "bold": True, "line_spacing": 1.02},
    ])
    stages = ["video", "perception", "TRACKING", "TEMPORAL FEATURES", "12 BEHAVIOURS", "EVENT GRAPH", "risk", "incident", "console"]
    x = MARGIN
    for stage in stages:
        hot = stage.isupper()
        w = Inches(0.62 + 0.098 * len(stage))
        box = s.shapes.add_shape(1, x, Inches(2.25), w, Inches(0.42))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xF6, 0xEE, 0xD9) if hot else PAPER
        box.line.color.rgb = HAZARD if hot else RULE
        box.line.width = Pt(1)
        box.shadow.inherit = False
        tf = box.text_frame
        tf.margin_left = tf.margin_right = Inches(0.04)
        run = tf.paragraphs[0].add_run()
        run.text = stage.lower()
        run.font.name, run.font.size = MONO, Pt(10)
        run.font.bold = hot
        run.font.color.rgb = INK if hot else INK2
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        x = x + w
    _text(s, MARGIN, Inches(2.95), Inches(11.3), Inches(0.7), [
        {"text": "The highlighted stages decide whether a movement was a drop, a throw, a drag or a careful "
                 "placement. They cost under one millisecond per frame — detection is ~99% of the pipeline, so "
                 "the part that differentiates this system is effectively free.",
         "size": 13, "color": INK2, "line_spacing": 1.4},
    ])
    points = [
        ("THRESHOLDS IN OBJECT-HEIGHTS",
         "A box falling 1.5× its own height fell the same amount whether the camera is 3 m or 8 m away. "
         "Pixel thresholds silently stop firing the moment a camera moves — and nobody notices, because "
         "the failure is silence."),
        ("RISK AND CONFIDENCE NEVER MERGE",
         "Risk is how bad if this is real. Confidence is how sure we are it is real. Multiplying them hides "
         "the case that matters most: high risk, low confidence — a review queue, not an alarm."),
        ("EVERY DETECTOR HAS A HARD NEGATIVE",
         "Gentle placement must not fire drop. Carrying at knee height must not fire drag. A detector that "
         "fires on everything passes every positive test and destroys the demo."),
    ]
    for i, (head, body) in enumerate(points):
        x = MARGIN + Inches(4.0) * i
        _text(s, x, Inches(4.15), Inches(3.6), Inches(0.3), [
            {"text": head, "font": MONO, "size": 9.5, "bold": True, "color": HAZARD},
        ])
        _rule(s, Inches(4.44), left=x, width=Inches(3.6))
        _text(s, x, Inches(4.6), Inches(3.6), Inches(2.2), [
            {"text": body, "size": 12, "color": INK2, "line_spacing": 1.4},
        ])

    # ---- 03 built ---------------------------------------------------------
    s = _blank(prs)
    _slug(s, "BUILT", "WHAT ACTUALLY RUNS", "03 / 06")
    _text(s, MARGIN, Inches(1.15), Inches(9.5), Inches(0.8), [
        {"text": "Twelve behaviours, one offline command.",
         "font": DISPLAY, "size": 33, "bold": True, "line_spacing": 1.02},
    ])
    points = [
        ("DETECTION WITHOUT TRAINING",
         "YOLO-World open-vocabulary: warehouse classes come from text prompts in a config file, so the "
         "prompt strings are a tuning surface — as much a threshold as anything else."),
        ("OFFLINE BY CONSTRUCTION",
         "Weights committed; CLIP ships as four 90 MiB chunks reassembled with a SHA256 check. Verified "
         "with socket.connect patched to raise — zero outbound connections."),
        ("BEHAVIOUR, NOT IDENTITY",
         "Evidence clips blur the upper quarter of every person box. Deliberately not face detection — that "
         "would build the exact capability we say we don't have."),
    ]
    for i, (head, body) in enumerate(points):
        x = MARGIN + Inches(4.0) * i
        _text(s, x, Inches(2.1), Inches(3.6), Inches(0.3), [
            {"text": head, "font": MONO, "size": 9.5, "bold": True, "color": HAZARD},
        ])
        _rule(s, Inches(2.39), left=x, width=Inches(3.6))
        _text(s, x, Inches(2.55), Inches(3.6), Inches(1.6), [
            {"text": body, "size": 12, "color": INK2, "line_spacing": 1.4},
        ])
    _table(s, MARGIN, Inches(4.35), W - MARGIN * 2,
           [["TIER", "BEHAVIOURS", "WHAT THE TIER MEANS"],
            ["Strong", "B01 drop · B05 improper stack · B06 unstable stack · B07 zone",
             "Held-out precision and recall 1.000 on synthetic reasoning tests"],
            ["Moderate", "B02 throw · B03 drag · B09 stepping · B12 unsafe surface",
             "Fires, with named failure modes we can describe"],
            ["Lightly validated", "B04 rough handling · B10 heavy handling · B11 unsafe sequence",
             "Proxy heuristics with confounds we can name, so we name them"]],
           [Inches(1.9), Inches(5.5), Inches(4.5)],
           caption="BEHAVIOURS, TIERED BY THE EVIDENCE BEHIND THEM — NOT BY AMBITION")
    _text(s, MARGIN, Inches(6.5), Inches(11.4), Inches(0.6), [
        {"text": "B10 is called “large item handled without equipment present”, not “unsafe lift”. It cannot "
                 "see mass, and mass is the whole concept, so the name says what was observed rather than "
                 "what was inferred.", "size": 12, "color": INK2, "line_spacing": 1.35},
    ])

    # ---- 04 evidence ------------------------------------------------------
    s = _blank(prs)
    _slug(s, "EVIDENCE", "ABLATION, HELD-OUT SPLIT", "04 / 06")
    _text(s, MARGIN, Inches(1.15), Inches(9.5), Inches(0.8), [
        {"text": "Remove tracking and the system goes completely silent.",
         "font": DISPLAY, "size": 33, "bold": True, "line_spacing": 1.02},
    ])
    _text(s, MARGIN, Inches(2.1), Inches(11.4), Inches(0.7), [
        {"text": "Measured on physics-rendered clips with the renderer's exact box geometry injected as "
                 "perception, so this isolates the reasoning layer. Two splits: a tune set of 9 clips "
                 "development happened against, and a heldout set of 27 drawn separately — different release "
                 "heights, launch speeds, stack offsets, and carton sizes from 60 to 100 px.",
         "size": 13, "color": INK2, "line_spacing": 1.4},
    ])
    _table(s, MARGIN, Inches(3.15), W - MARGIN * 2,
           [["VARIANT", "TUNE (N=9)", "HELDOUT (N=27)", "READING"],
            ["baseline", "1.000", "0.933", "The gap between columns is the size of the overfit"],
            ["no_tracking", "0.000", "0.000", "Nothing fires at all. Every behaviour is defined over a sequence."],
            ["no_smoothing", "1.000", "0.867", "Costs nothing on fixed clips; costs 0.066 once sizes and speeds vary"],
            ["no_event_graph", "0.933", "0.933", "No delta — it feeds the risk score, not event detection. Reported anyway."]],
           [Inches(2.2), Inches(1.5), Inches(1.8), Inches(6.4)],
           caption="REASONING-LAYER F1 — MECHANISMS SWITCHED OFF ONE AT A TIME", key_row=2)
    for i, (head, body) in enumerate([
        ("WHY THE TUNE COLUMN READS 1.000",
         "Because thresholds were changed in response to failures on those nine clips. A perfect score on "
         "your own tuning set is a warning, not a result — so the held-out split exists, and 0.933 is the "
         "number we quote."),
        ("WHAT A HELD-OUT SPLIT DOESN'T FIX",
         "Both splits come from the same renderer. A systematic error in how it models handling appears "
         "identically in both. Only real footage closes that gap — which is the next slide."),
    ]):
        x = MARGIN + Inches(6.05) * i
        _text(s, x, Inches(5.55), Inches(5.6), Inches(0.3), [
            {"text": head, "font": MONO, "size": 9.5, "bold": True, "color": HAZARD},
        ])
        _rule(s, Inches(5.84), left=x, width=Inches(5.6))
        _text(s, x, Inches(6.0), Inches(5.6), Inches(1.2), [
            {"text": body, "size": 12, "color": INK2, "line_spacing": 1.4},
        ])

    # ---- 05 reality check -------------------------------------------------
    s = _blank(prs)
    _slug(s, "REALITY CHECK", "SESSION 1 — REAL LOADING BAY", "05 / 06")
    # Sized to stay on one line even when Archivo falls back to a wider face —
    # a two-line title here collides with the rule under it.
    _text(s, MARGIN, Inches(1.05), Inches(12.0), Inches(0.8), [
        {"text": "Then we pointed it at a real dock, and it did badly.",
         "font": DISPLAY, "size": 29, "bold": True, "line_spacing": 1.02},
    ])
    _rule(s, Inches(2.0), colour=CRIT, height=Inches(0.035))
    _text(s, MARGIN, Inches(2.12), Inches(2.6), Inches(1.0), [
        {"text": "1/9", "font": DISPLAY, "size": 60, "bold": True, "color": CRIT, "line_spacing": 0.9},
    ])
    _text(s, MARGIN + Inches(2.5), Inches(2.2), Inches(8.9), Inches(0.9), [
        {"text": "expected behaviours reported across seven clips of genuine loading-bay operations. Not "
                 "silent — incidents on 5 of 7 clips — but mostly the two loosest detectors rather than the "
                 "behaviour each clip was recorded to show.", "size": 13, "color": INK2, "line_spacing": 1.35},
    ])
    _rule(s, Inches(3.3))
    _text(s, MARGIN, Inches(3.45), Inches(11.4), Inches(0.3), [
        {"text": "This is the number we lead with, because everything on the previous slide was measured on "
                 "clips that could not fail in these ways. One afternoon of real footage found four defects:",
         "size": 12.5, "color": INK2},
    ])
    findings = [
        ("FIXED", GOOD, "A config knob was declared and read by nothing.",
         "min_track_age_frames sat in the config, documented, wired to nothing. A one-frame-old track's first "
         "velocity is a full-magnitude jump — on soft footage that is jitter, and it fired throw on clips of "
         "people dragging."),
        ("FIXED", GOOD, "A single horizontal floor line cannot describe a perspective view.",
         "One line puts a box on the floor at one depth and a metre up at another — why drag never fired on "
         "any of four dragging clips. The nearest person's feet are now the ground reference."),
        ("FIXED", GOOD, "“Held by a person” was true in 100% of frames.",
         "Overlap plus any vertical intersection means “somebody is standing behind it”. A thrown carton read "
         "as held throughout, so the throw detector could not fire by construction."),
        ("NOT FIXABLE BY TUNING", CRIT, "Throw and drag are not separable on this footage.",
         "Vertical acceleration p99 is ~48 object-heights/s² against a free-fall value of ~20 — below the noise "
         "floor of box jitter. A ballistic test was designed, then not built: measuring first showed it could "
         "not work."),
    ]
    y = Inches(3.92)
    for tag, colour, what, why in findings:
        _rule(s, y, colour=colour, height=Inches(0.68), left=MARGIN, width=Emu(28575))
        _text(s, MARGIN + Inches(0.14), y - Inches(0.02), Inches(12.0), Inches(0.72), [
            {"text": tag, "font": MONO, "size": 8, "bold": True, "color": colour},
            {"text": what, "size": 12, "bold": True, "space_before": 1},
            {"text": why, "size": 10, "color": INK2, "space_before": 1, "line_spacing": 1.2},
        ])
        y = y + Inches(0.8)
    _text(s, MARGIN, Inches(7.14), Inches(12.0), Inches(0.3), [
        {"text": "Precision on real footage is unmeasurable, and labelled that way: session 1 has no "
                 "hard-negative clip, so every clip is a positive and a detector that fired constantly would "
                 "score perfectly.", "size": 10.5, "color": INK2},
    ])

    # ---- 06 next ----------------------------------------------------------
    s = _blank(prs)
    _slug(s, "NEXT", "ORDERED BY VALUE PER HOUR", "06 / 06")
    _text(s, MARGIN, Inches(1.15), Inches(10.5), Inches(0.8), [
        {"text": "Four things move this further than any amount of tuning.",
         "font": DISPLAY, "size": 33, "bold": True, "line_spacing": 1.02},
    ])
    asks = [
        ("Sixty seconds of ordinary handling",
         "One clip where nothing bad happens. Without it, precision on real footage stays permanently "
         "unmeasurable and the alarm rates cannot be interpreted at all. Cheapest item here, highest value."),
        ("Start and end times logged at record time",
         "Turns a clip-level presence check into real precision and recall. Reconstructing spans afterwards "
         "means scoring the system against labels drawn after seeing what the system did."),
        ("Direct camera export",
         "Session 1 is a phone recording of a CCTV playback window. Exporting from the recorder removes two "
         "lossy encodes, the screen moiré, the software chrome and the drawn-on annotations in one step."),
        ("Closer framing",
         "A carton is currently ~200 px tall. Raise that and the acceleration signal rises above box jitter — "
         "the one thing standing between throw and drag being separable at all."),
    ]
    y = Inches(2.2)
    for i, (head, body) in enumerate(asks, start=1):
        _text(s, MARGIN, y, Inches(0.5), Inches(0.3), [
            {"text": f"{i:02d}", "font": MONO, "size": 11, "bold": True, "color": HAZARD},
        ])
        _text(s, MARGIN + Inches(0.55), y - Inches(0.03), Inches(10.8), Inches(0.9), [
            {"text": head, "size": 14, "bold": True},
            {"text": body, "size": 11.5, "color": INK2, "space_before": 2, "line_spacing": 1.3},
        ])
        y = y + Inches(1.02)
        _rule(s, y - Inches(0.14), colour=RGBColor(0xDE, 0xE6, 0xE8))
    _text(s, MARGIN, Inches(6.55), Inches(11.4), Inches(0.4), [
        {"text": "Every number here is traceable to a file: the claims ledger in STATE.md lists each one, "
                 "where it was measured, and which are still marked not measured.",
         "size": 11.5, "color": INK2},
    ])
    _rule(s, Inches(7.02))
    _text(s, MARGIN, Inches(7.14), Inches(11.4), Inches(0.3), [
        {"text": "TEAM BAY 9    ·    SPOC ANIRUDH BADAMPUDI    ·    DOCKSENSE    ·    HACKATHON PROTOTYPE",
         "font": MONO, "size": 9, "color": INK3},
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("submission/GEGAI_Bay9/DockSense_Concept_Deck.pptx"))
    args = ap.parse_args()
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
