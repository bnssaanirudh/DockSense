# DockSense

**AI video intelligence for warehouse handling.** Turns loading-bay video into
explainable, reviewable incidents — what happened, which entity, when, why it was
judged risky, how confident the system is, and what to do about it.

> **Status: hackathon prototype.** Read [Honest scope](#honest-scope) before
> quoting any capability. Behaviours differ in how well they are validated, and
> that difference is stated rather than smoothed over.

---

## The problem

Warehouses already have cameras. Conventional CCTV produces **evidence after
damage**, not prevention before it: footage gets reviewed once a claim is filed,
by which point the product is broken, the cause is disputed, and the handling
pattern that caused it has repeated unnoticed for weeks.

The gap is not recording. It is that nobody watches hours of footage, and **a
single frame cannot tell you whether a box was placed or dropped**. That
distinction lives in a sequence.

## The approach

Rather than asking a model to label each frame safe/unsafe, DockSense tracks
entities and reasons over short temporal histories — what moved, how fast,
whether contact happened, whether support geometry became unstable, and what
happened either side of the event.

```
video → perception → tracking → features → behaviours → events → risk → incidents → db → api → web
                                                                             ↘ assistant ↗
```

Strictly one-directional; the CV core imports nothing from the database or web
layers, and a test enforces it.

Two design choices worth calling out:

- **Thresholds are in object-heights, not pixels.** A box falling 1.5× its own
  height fell the same amount whether the camera is 3 m or 8 m away. Pixel
  thresholds silently stop firing the moment a camera moves.
- **Risk and confidence are separate fields, never merged.** Risk is "how bad if
  this is real". Confidence is "how sure are we it is real". A high-risk,
  low-confidence event belongs in a review queue, not in an alarm.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/setup_offline.py     # reassembles CLIP weights from committed chunks
./scripts/demo.sh                   # API + console on http://127.0.0.1:8000
```

`./scripts/demo.sh --check` runs the preflight without starting anything.

**The demo runs with the network disabled.** Detector weights are committed;
CLIP ViT-B-32 ships as four ~90 MiB chunks (it is 338 MB, over GitHub's 100 MB
per-file limit) and is reassembled locally with a SHA256 check. Verified by
running the pipeline with `socket.connect` patched to raise: completes on real
CCTV in 8.3 s with zero outbound connections.

### Data (not in git)

Video is gitignored. After cloning:

```bash
python scripts/fetch_public_data.py   # ~1.9 GB public CCTV, CC BY 4.0
python scripts/render_synthetic.py    # deterministic synthetic physics clips
```

### Run the pipeline

```bash
python -c "from handleguard.pipeline import run; \
  print(run('data/public/tune/walkway_violation/0_tr1.mp4', write_clips=False))"

python scripts/run_ablations.py --videos data/synthetic/*.mp4 \
    --ground-truth data/synthetic/ground_truth.csv
python scripts/evaluate_events.py --predictions-db handleguard.db
pytest -q
```

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/incidents` | list, with `behaviour_id` / `name` / `min_risk` / `band` / `review_status` / `zone` / `limit` filters |
| GET | `/incidents/{id}` | one incident with full evidence |
| PATCH | `/incidents/{id}/review` | supervisor disposition + note |
| GET | `/stats` | totals and per-behaviour counts |
| GET | `/clips/{file}` | evidence clip or thumbnail |
| POST | `/chat` | grounded assistant |

The built console is served from `/` by the same process, so the demo is one
command with no second server and no CORS hop.

---

## Behaviours

All twelve are implemented. They are **not equally trustworthy**, and the table
says which is which.

| ID | Behaviour | Basis | Confidence in it |
|---|---|---|---|
| B01 | Product dropped | Vertical kinematics + impact deceleration | Strong |
| B02 | Product thrown | Horizontal velocity while unsupported | Strong |
| B05 | Improper stack (large on small) | Two-box area + overlap geometry | Strong |
| B06 | Unstable stack | Support ratio below threshold, sustained | Strong |
| B07 | Product outside designated zone | Zone polygon + dwell time | Strong |
| B08 | Pallet overhang | Product footprint vs pallet footprint | Strong |
| B03 | Product dragged | Floor proximity + horizontal travel | Moderate |
| B09 | Stepping on product | Box contact geometry, no pose model | Moderate |
| B12 | Unsafe-surface zone | Operator-configured zone, not visual | Moderate |
| B04 | Rough handling | Acceleration proxy — sensitive to tracker jitter, overlaps B01/B02 | **Lightly validated** |
| B10 | Large item handled without equipment present | Size proxy; **weight is not observable from video** | **Lightly validated** |
| B11 | Unsafe loading sequence | Ordered event pairs; inherits all upstream error | **Lightly validated** |

The three *lightly validated* rows are proxy heuristics with confounds we can
name, so we name them. B04 suppresses itself when B01/B02 already claimed the
entity, and B10 goes silent as soon as handling equipment is visible — both to
stop them generating noise they cannot justify.

Each implemented detector has a positive test **and a named hard negative** —
gentle placement must not fire drop, carrying at knee height must not fire drag,
a correct small-on-large stack must not fire improper stack. The negative is the
one that matters: a detector that fires on everything passes every positive test
and destroys the demo.

---

## Honest scope

**What is verified.** Measured on Apple M3 / 8 GB / MPS, 1920×1080 CCTV
downscaled to 1280×720, `imgsz=640`:

| Metric | Value |
|---|---|
| Detection latency p50 / p95 | **28.2 ms / 44.8 ms** (warmup excluded) |
| End-to-end throughput | **21.1 fps** (offline batch, not real-time) |
| Tracking + features + behaviours | **< 1 ms combined** |
| Detections on real CCTV | 57 and 52 on sample frames, correctly classed |
| Offline operation | verified with sockets blocked — zero outbound connections |
| Tests | 140 passing |

Full breakdown in `artifacts/evaluation/latency.json`. Detection is ~99% of
pipeline time, which means **the temporal reasoning layer is effectively free** —
the part that differentiates this system costs under a millisecond a frame.

**Reasoning layer, measured.** On physics-rendered clips with exact box geometry
injected as perception (5 positives, 4 hard negatives, temporal IoU 0.3):

| Variant | Precision | Recall | F1 |
|---|---:|---:|---:|
| baseline | 0.750 | 0.600 | **0.667** |
| `no_tracking` | 0.000 | 0.000 | **0.000** |
| `no_smoothing` | 0.750 | 0.600 | 0.667 |
| `no_event_graph` | 0.750 | 0.600 | 0.667 |

**`no_tracking` collapsing to zero is the evidence for the central claim**: remove
persistent identity and nothing temporal accumulates, so no behaviour fires at all.
The system detects sequences, not frames.

`no_smoothing` and `no_event_graph` show **no delta**, and that is reported rather
than hidden. These clips have zero detector jitter for smoothing to remove, and the
event graph feeds the *risk score* rather than event detection, so event F1 is the
wrong instrument for it. A flag that changes nothing is worth knowing about.

Two caveats that travel with these numbers: this measures **reasoning only**
(perception is held perfect by construction), and the **thresholds were tuned on
this set, so it is not held out**. Full report: `artifacts/evaluation/reasoning_eval.md`.

**What is still not measured.** Per-behaviour precision/recall on *real* footage.
Ground-truth video of drops, throws and stacking has not been recorded, so no
field accuracy number should appear anywhere.

**Why the public dataset does not fill that gap.** The Unsafe-Net footage is a
metal-press factory, and its *Safe Walkway Violation* label means *a person off a
marked pedestrian walkway*. B07 means *a product in a restricted loading-bay
zone* — different subject, and inverted zone semantics (there the walkway is the
safe area you stay inside; here the zone is the forbidden area you stay out of).
Drawing polygons until B07 fired on those clips would yield a precision/recall
number that looks real but measures a behaviour we did not build. So the public
data backs **detector validation** and **hard negatives** only.

**Not claimed:** real-time inference, multi-camera tracking, worker
identification, confirmed product damage, any trained model, or absolute metric
measurement — there is no camera calibration, so every quantity is
object-relative by design.

---

## Responsible AI

- **Behaviour, not identity.** No face recognition, no re-identification, no
  worker names, no ranking. Stored evidence clips blur the upper 25% of every
  detected person box (`configs/behaviours.yaml` → `privacy.blur_faces`,
  on by default). Deliberately *not* face detection: running a face detector to
  decide what to blur would build the exact capability we say we do not have,
  and it fails open — an undetected face is an unblurred face. This **reduces
  identifiability; it is not anonymisation** — gait, clothing and context
  remain.
- **Observed / inferred / confirmed are kept distinct.** The system reports what
  it observed and what risk it inferred. It never claims damage occurred.
- **No intent inference.** B10 is named *"large item handled without equipment
  present"* — an observation, not a judgment. It cannot see mass, and mass is the
  whole concept, so it says so.
- **B05 uses a size proxy, and says so.** Video cannot observe weight; the
  evidence dict carries `"basis": "size proxy from bounding boxes, not measured
  weight"`.
- **Human review gates consequence.** Alerts are decision support. The system
  produces no disciplinary output.
- **The assistant cannot invent events.** It answers only from stored incidents,
  cites incident IDs, says *"No matching incidents found."* on an empty result,
  and refuses identity and ranking questions before any model call.

---

## Datasets and attribution

Both datasets are **CC BY 4.0**, which requires attribution.

**Safe and Unsafe Behaviours (Unsafe-Net)** — Önal, O. & Dandıl, E. (2024),
*Video dataset for the detection of safe and unsafe behaviours in workplaces*,
Data in Brief. https://data.mendeley.com/datasets/xjmtb22pff/1 —
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Used here for detector validation and hard negatives.

**LOCO — Logistics Objects in Context** — Mayershofer et al., Technical
University of Munich. https://github.com/tum-fml/loco — licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used for validating
open-vocabulary detector prompts against real logistics scenes.

Neither dataset was modified in place; subsets are fetched by
`scripts/fetch_public_data.py`, which records provenance.

---

## Stack

Python 3.13 · PyTorch 2.10 (MPS) · Ultralytics YOLO-World (open-vocabulary, zero
training) · ByteTrack · OpenCV · FastAPI · SQLite (stdlib `sqlite3`) · React + Vite

No model is trained. Warehouse classes come from text prompts in
`configs/products.yaml`, which makes prompt strings a tuning surface — they are
as much a threshold as anything in `configs/behaviours.yaml`.

## Limitations

- Track ID switches break temporal behaviours. Windows are kept short (≤2 s) so a
  switch costs one event rather than all of them, but the failure mode is real.
- No camera calibration, so no absolute distances or speeds.
- Zone polygons are per-camera and must be redrawn when a camera moves.
- Synthetic clips validate behaviour *logic* only. YOLO-World returns zero
  detections on them, correctly — it is trained on photographs and a flat grey
  rectangle is not a cardboard box. Perception is validated on real footage.

## Repository

`handleguard/` pipeline · `apps/api` FastAPI · `apps/web` React console ·
`configs/` thresholds and zones · `scripts/` setup, fetch, evaluation, demo ·
[`plan.md`](plan.md) architecture and sequencing · [`STATE.md`](STATE.md) live
status and handoff · [`project.md`](project.md) requirements and acceptance criteria
