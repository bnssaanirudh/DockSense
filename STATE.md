# STATE — DockSense

Relay handoff file. **Read this first, update it last.**

---

## Protocol

1. `git pull`
2. Read this file top to bottom.
3. Pick the top unblocked item from **Next tasks**.
4. Build it. Run the check listed with it.
5. Update **Recent changes**, **Next tasks**, **Blockers**. Move done items to **Done**.
6. `git add -A && git commit && git push`

Rules:
- One person per owned directory at a time (see **Ownership**). Stay in your lane, no merge hell.
- Never mark a task done without running its check.
- If you tune a threshold, write down **which session's footage you tuned on**. S3 is off-limits.
- Do not put a number in the README or deck that you did not personally run.

---

## Ownership

Assigned 8 Sep from demonstrated work in git history, not guessed.

| Lane | Directories | Owner |
|---|---|---|
| **A — CV / perception / infra** | `handleguard/{video,perception,tracking,features}`, `scripts/`, `models/`, `configs/` | **Akshat** (`akshatagrawal.work@gmail.com`) |
| **B — Reasoning / evaluation** | `handleguard/{behaviours,events,risk,incidents,evaluation}`, `tests/` | **Anirudh** (`anirudhbadampudi@gmail.com`) |
| **C — Product / demo** | `apps/api/`, `apps/web/`, `handleguard/{db,assistant}`, `artifacts/`, slides | **UNASSIGNED — third teammate, name needed** |

Shared, coordinate before editing: `configs/`, `STATE.md`, `TASK_SHEET.md`, `requirements.txt`.

**Note:** git history shows only two contributors. Lane C is currently
uncovered, and it owns the demo, screenshots and slides — i.e. everything the
judges actually see. Until the third person is named, **Lane C work is split:
Anirudh takes `apps/web` (already active there), Akshat takes `apps/api` +
demo packaging.**

### Remaining work is no longer three parallel lanes

All twelve behaviours are implemented and the code is largely complete. What is
left is mostly sequential and mostly gated on footage:

| Work | Owner | Gated on |
|---|---|---|
| **Film S1/S2/S3** | Team — whoever is free first | Nothing. **This is the critical path.** |
| Tune thresholds on S1/S2 | A + B together | Footage |
| S3 held-out evaluation (run **once**) | B | Footage + tuning |
| Ablation table on real footage | B | Footage |
| Latency p50/p95 instrumentation | A | Nothing |
| Screenshots, deck, demo recording | C | A working demo |
| Two offline rehearsals | Team | Everything else |

---

## Deadline

**10 September 2026 — TODAY.** Submission day.

---

## Current state

Plan approved ([plan.md](plan.md)), GATE 1 signed. Foundation and code-owned P0
landed, committed, and pushed through `e82f958`. P1 now has a deterministic
synthetic vertical slice:

- `handleguard/types.py` — **FROZEN.** Shared dataclasses. Do not edit without announcing here.
- `handleguard/config.py` — YAML entry point.
- `handleguard/perception/geometry.py` — pure geometry, 13 tests.
- `scripts/fetch_public_data.py` — CC BY 4.0 subset fetch (reproducible).
- `scripts/render_synthetic.py` — Newtonian synthetic drop/throw/drag/place clips
  with frame-exact ground truth. Lets behaviour detectors be tuned + measured
  before any real footage exists.
- `models/yolov8s-worldv2.pt` — committed (25 MB), MPS benchmarked at 36 fps.

- `handleguard/features/compute.py` — normalized velocities, acceleration, zone,
  floor proximity, support, and held-by-person features.
- `handleguard/events/dedup.py` — continuous detector firings collapse into one
  event per behaviour/track/window.
- `handleguard/risk/` — contextual risk scoring and guarded explanations.
- `handleguard/incidents/` — SOP-backed incident builder.
- `handleguard/pipeline.py` — video -> detect -> track -> features -> behaviours
  -> dedupe -> risk -> incident -> evidence media -> optional DB.
- `apps/api/main.py` — FastAPI read/review/chat/clip endpoints over SQLite.
- `handleguard/assistant/templates.py` — offline deterministic assistant answers
  with guardrails, citations, and identity refusal.
- `apps/web/` — Vite/React incident review console over the API.
- `handleguard/evaluation/metrics.py` + `scripts/evaluate_events.py` — temporal-IoU
  event evaluation with per-behaviour TP/FP/FN, precision, recall, F1, and n.

Added 8 Sep:

- **All 12 behaviour detectors implemented, zero stubs.** Each has a positive test
  and a named hard negative.
- `handleguard/events/graph.py` — the Temporal Event Graph. Relates deduplicated
  events by FOLLOWS / SHARES_TRACK / RECURS; `chains()` gives per-entity stories,
  `recurrence()` feeds the frequency risk component so a third drop of the same
  carton outscores the first.
- `PipelineFlags` + `scripts/run_ablations.py` — the ablation switches are now
  wired to real mechanisms and executable end to end.
- `models/clip_parts/` + `scripts/setup_offline.py` + `scripts/demo.sh` — offline
  operation, verified with sockets blocked.
- `README.md` — with the CC BY 4.0 attribution both datasets require.

Added 9 Sep (full audit pass):

- `handleguard/privacy/` — **was an empty directory while README, project.md and
  plan.md all claimed face blurring.** Now implemented and wired into the evidence
  clip writer, on by default via `privacy.blur_faces`.
- Per-stage latency instrumentation with real measured numbers.
- Console served from the API at `/`; demo is one offline command.

Added 10 Sep:

- `scripts/eval_reasoning.py` + exact per-frame boxes from the renderer — the
  reasoning layer is now **measured**, not asserted. First run scored F1 0.222 and
  exposed that **B01 never fired on an actual drop**: at 8 fps a falling carton
  outruns IoU association and the tracker loses it mid-fall. After the fixes,
  **F1 0.667**, with `no_tracking` collapsing to **0.000**.
- `artifacts/evaluation/reasoning_eval.md` — results with the caveats attached.

**140 tests pass.**

**Still missing: real footage of drops / throws / stacking.** The reasoning layer
is now measured on rendered clips and perception is verified on real CCTV, but
**no real video exists where the two meet**. Every "robustly demonstrated" claim
still depends on that.

---

## Data is NOT in git — fetch it after cloning

`data/raw/`, `data/public/`, `data/synthetic/`, `data/processed/`, `data/clips/`
are all gitignored. A fresh clone has no video. To reproduce:

```bash
pip install -r requirements.txt
python scripts/fetch_public_data.py      # ~1.9 GB, CC BY 4.0, into data/public/
python scripts/render_synthetic.py       # deterministic, into data/synthetic/
```

`data/raw/` = our own recordings (S1/S2/S3). Whoever films uploads them to shared
storage out of band; they are never committed (size + privacy).

---

## Blockers

| Blocker | Severity | Owner | Note |
|---|---|---|---|
| **No real footage of drop / throw / stacking** | CRITICAL | unassigned | Public CCTV covers B07 + hard negatives only. B01/B02/B05/B06/B08 have no real video to fire on. **Decision taken 8 Sep: we record.** Full brief below — see *Recording brief*. ~35 min. |
| **No behaviour has ever fired on real video** | HIGH | CV | Reasoning is now measured on rendered clips (F1 0.667) and detection on real CCTV, but the two have never met on real footage. Zones are still placeholders. |
| **Lane C has no owner** | HIGH | team | Git history shows only two contributors. Lane C owns the demo, screenshots and slides — everything the judges actually see. Interim split recorded in Ownership; name the third person or accept the split. |

### Closed 8 Sep

| Was | Finding |
|---|---|
| ~~"Offline demo broken — 338 MB CLIP downloads at runtime"~~ | **Fixed.** CLIP ships as four ~90 MiB chunks in `models/clip_parts/` (338 MB is over GitHub's 100 MB per-file limit; LFS would add tooling plus a 1 GB/month cap ≈ 3 clones). `scripts/setup_offline.py` reassembles with a SHA256 check; `scripts/demo.sh` reassembles then hard-fails with a fix message rather than silently downloading mid-demo. **Verified with `socket.connect` patched to raise: pipeline completes on real CCTV in 8.3 s with zero outbound connections.** |
| ~~"8 of 12 behaviours are stubs"~~ | **All 12 implemented, zero stubs.** B05/B06/B08/B09/B12 then B04/B10/B11 added 8 Sep, each with a positive test and a named hard negative. B04 and B10 carry suppression logic (defer to B01/B02; go silent when equipment is visible) so the weak three do not generate noise. |
| ~~"YOLO-World prompt validation not passing — 0 detections even at 0.01 conf"~~ | **Misdiagnosed. Prompts are fine.** Verified on real CCTV: `7_tr1.mp4` → **57 detections**, `4_te4.mp4` → **52**, correct classes (person / cardboard box / hand trolley). The 0-detection result happens **only on synthetic clips**, and it is expected and unfixable: `render_synthetic.py` draws flat coloured rectangles, and a model trained on photographs correctly refuses to call a grey rectangle a cardboard box. **Synthetic clips validate behaviour LOGIC via injected tracks; they can never validate perception.** Do not spend time tuning prompts. |

---

## Data semantics — why the public dataset is NOT B07 ground truth

Investigated 8 Sep. Worth reading before anyone tries to "just tune zones until
B07 fires on the public clips."

The Unsafe-Net footage is a **metal-press factory**, and its *Safe Walkway
Violation* label means **a person walking off a marked pedestrian walkway**.
Our B07 means **a product placed in a restricted zone in a loading bay**.
Different subject (person vs product) and inverted zone semantics (there, the
walkway is the *safe* area you must stay inside; here, the zone is the *forbidden*
area you must stay out of).

Traced the painted floor markings by HSV colour to check: the large green region
is a floor-marked **storage bay beside the shelving**, not the pedestrian route.

**If we drew polygons until B07 fired on their clips, we would produce a
precision/recall number that looks real but measures a behaviour we did not
build.** A judge probing the eval would find it. So:

| Public data IS used for | Public data is NOT used for |
|---|---|
| Detector validation on real industrial video (**verified: 57 and 52 correct detections**, classes person / cardboard box / hand trolley) | B07 precision/recall |
| Hard negatives — "does the system stay silent on normal operation" | Any per-behaviour metric |
| Demo b-roll showing real-world footage | The "robustly demonstrated" claims |

Real zone metrics come from **our own footage**, where we tape the zones and
therefore control what they mean.

---

## Recording brief — READ THIS BEFORE FILMING

Decision taken 8 Sep: **we record.** Without this footage, five of the six flagship
behaviours (B01 drop, B02 throw, B05 improper stack, B06 unstable stack, B08 pallet
overhang) have no real video to fire on, and the submission cannot claim they were
demonstrated. Public CCTV only covers B07 and hard negatives.

**Time: ~35 minutes. Two people. No warehouse needed.**

### Kit

| Item | Notes |
|---|---|
| 8–12 cardboard boxes | Amazon/delivery boxes fine. Need **at least 3 clearly large** and **5 small** — B05 needs a visible size difference |
| 1 pallet substitute | Real pallet ideal. Otherwise a low wooden board, crate, or upturned tray. Note in `takes.csv` what you used |
| 1 trolley substitute | Hand truck, luggage trolley, or office chair. Must be visibly *equipment*, not a box |
| Masking tape | Mark zone boundaries on the floor |
| Marker pen | **Write a big number on each box.** Massively helps tracking and annotation later |
| Phone + tripod | Or prop it on a stack of books. **Never hand-hold** — camera shake creates fake velocity and fires false drops |

### Camera setup

- **Fixed position.** Does not move at all within a session.
- Landscape, **1080p, 30 fps**.
- Frame must contain: the **floor line**, the **pallet**, and the **full height a box is lifted to**. If the box leaves frame at the top, the drop is unmeasurable.
- Good even light. No window or lamp directly behind the scene.
- Tape two floor zones and note which is which:
  - `staging` — products allowed
  - `walkway` — products forbidden (this is what B07 fires on)

### Session structure — this part matters most

Record in **sessions**. A session = one unbroken camera position.
Name every file `S{n}_{behaviour}_{take}.mp4` → e.g. `S1_drop_03.mp4`.

| Session | Camera | Purpose |
|---|---|---|
| **S1** | Near, side-on, ~3 m | Primary tuning data |
| **S2** | Far, side-on, ~6 m | Scale robustness |
| **S3** | Elevated / angled ~30° | **HELD OUT. Never tuned on. Evaluated once, at the end.** |

**Why sessions and not clips:** the train/test split is by *session*. Two takes of the
same drop from the same camera position must never land on opposite sides of the split
— that's leakage, and it's the first thing a judge probes. Recording in labelled
sessions is what makes an honest split possible at all.

S3 is the honesty artifact. Tuning on S1/S2 and reporting on S3 is the difference
between a real number and one that gets dismantled.

### Shot list — 3 takes minimum per behaviour per session

Vary speed and box size between takes.

| # | Behaviour | What to do | Priority |
|---|---|---|---|
| B01 | Drop | Lift box to chest height, release cleanly, let it hit the floor | **FLAGSHIP** |
| B02 | Throw | Toss box sideways 1–2 m onto floor or pallet | **FLAGSHIP** |
| B05 | Improper stack | Place a **large** box on top of a **small** one, leave it | **FLAGSHIP** |
| B06 | Unstable stack | Stack boxes with big overhang / visible lean | **FLAGSHIP** |
| B08 | Pallet overhang | Place box so ~half hangs off the pallet edge | **FLAGSHIP** |
| B03 | Drag | Push/pull box along floor 2 m+ without lifting | secondary |
| B07 | Zone violation | Place box in the taped `walkway` zone, leave 5 s+ | secondary |
| B09 | Stepping on box | Step on a box, hold foot there 2 s+ | secondary |
| B04 | Rough handling | Slam box down hard onto pallet; shove box into another | secondary |
| B10 | Manual heavy lift | Carry the largest box alone, **no trolley in frame** | secondary |
| B12 | Unsafe surface | Move box through the taped unsafe zone | secondary |
| B11 | Unsafe sequence | Lift heavy box, move, place unstably — trolley visible but unused | secondary |

**If short on time: shoot the five FLAGSHIP rows across S1 and S3 and stop.** Those are
what the submission's headline claims rest on.

### Hard negatives — do NOT skip

**Most-skipped, highest-value part of the shoot.** Without these, every detector looks
perfect because it never gets a chance to be wrong. These clips are what the
false-positive rate is measured on.

Record **5+ takes each**:

| Clip | Must NOT fire |
|---|---|
| Gentle controlled placement | drop |
| Carrying a box at knee height | drag |
| Box moved on the trolley | drag, manual-handling |
| Correct stack — small on large | improper stack |
| Person walking past a box, no contact | stepping |
| Box fully on pallet, well aligned | overhang |
| Person briefly crossing the walkway | zone violation (transient) |
| **Static scene, boxes at rest, 30 s** | anything at all |

That last one is the single best demo asset you will record. *"Here is 30 seconds of
normal operation and the system stayed silent"* answers the question every judge is
privately asking.

### Log every take — `data/raw/takes.csv`

One line per take, written **at record time**. Reconstructing timestamps from footage
afterwards takes hours; this takes seconds and is the input to every metric we report.

```csv
filename,session,behaviour,approx_start_s,approx_end_s,notes
S1_drop_01.mp4,S1,drop,2.4,3.1,large box chest height
S1_normal_01.mp4,S1,none,,,gentle placement
```

### Done when

- [ ] 3 sessions from genuinely different camera positions
- [ ] 5 flagship behaviours × 3 takes × at least S1 and S3
- [ ] 8 hard-negative categories, 5 takes each
- [ ] One 30 s+ fully-normal clip
- [ ] `takes.csv` filled in
- [ ] Files copied into `data/raw/` (gitignored — share via drive, never commit)

Full original version with extra detail: `docs/RECORDING_GUIDE.md`.

---

## Next tasks

Rewritten 9 Sep after a full file-wise and flow-wise audit. Blocks 1-4 of the
previous list are complete.

### THE ONLY CRITICAL-PATH ITEM

1. **Film S1/S2/S3.** ~35 min. See *Recording brief* below.
   Everything else is done or cosmetic. Without this the submission has **zero
   per-behaviour precision/recall**, and five of the six flagship behaviours have
   never fired on real video. This is the single thing standing between "built"
   and "measured", and measurement is what the judging rewards.

### Once footage exists (in order)

2. Draw zones on the real scene → `configs/zones.yaml`. *Check:* a clip produces ≥1 incident.
3. Tune thresholds on **S1/S2 only**. Log which session in the Decisions log.
4. Run S3 **exactly once**: `python scripts/evaluate_events.py`. Report as measured.
5. Ablation table on real video: `python scripts/run_ablations.py`.
6. Fill the claims ledger with whatever comes out, good or bad.

### Not gated on footage

7. Screenshots → `artifacts/screenshots/` (console runs now: `./scripts/demo.sh`).
8. 5-6 slide deck + demo recording.
9. **Two offline rehearsals with WiFi off.** One has been done from a clean clone;
   the second should be on the actual demo machine.
10. Name the third teammate, or accept the Lane C split in Ownership.

## Recent changes

| When | Who | What |
|---|---|---|
| 10 Sep | Claude | **Reasoning layer measured for the first time — and it was broken.** Built `scripts/eval_reasoning.py` (renderer now emits exact per-frame boxes, so reasoning is measured with perception held perfect). First run: **F1 0.222**. Root cause: a falling carton moves ~70 px/frame at 8 fps while being ~80 px tall, so ByteTrack lost the track mid-fall and **B01 never fired on an actual drop**. Fixes: inference_fps 8→24, match_thresh 0.8→0.95, drop velocity threshold 1.5→4.0 (controlled lowering peaks at 1.8, free fall at 13.6), horizontal share computed over the window instead of one frame, event start_t from motion onset instead of window start, and `settled()` exposing in-flight events so B04's suppression actually works. **F1 0.222 → 0.667.** Ablation: `no_tracking` collapses to 0.000. |
| 9 Sep | Claude | **Full audit — file-wise, flow-wise, backend, frontend, live browser.** Fixed: empty `privacy/` package despite three docs claiming face blurring (now implemented, wired into clip writer, tested on stored pixels); missing `metrics/__init__.py`; `seed_fake_incidents.py` ignoring `--help`; README stale test count and undocumented `PATCH /review`; STATE title still saying HandleGuard. Verified live: all 7 endpoints, path traversal blocked (404), invalid review status (422), console renders, filters work, review persists, assistant guardrail holds in the UI, zero console errors. Pruned 85 lines of stale task list. |
| 8 Sep | Claude | **All 12 behaviours implemented (B04/B10/B11 completed), ablation flags wired, temporal event graph added, README written.** 123 tests pass. B04 defers to B01/B02 and B10 goes silent when equipment is visible, so the weak three don't generate noise they can't justify. |
| 8 Sep | Claude | **B05/B06/B08/B09/B12 implemented — 4 behaviours to 9.** Shared geometry helpers added to `base.py` so detectors never touch raw `xyxy` (enforced by test). Fixed a real defect: `below()` used `is_above`'s default `min_overlap=0.3`, making a box overhanging >70% invisible to B06 — the most dangerous stack was the one it couldn't see. 103 tests pass. |
| 8 Sep | Claude | **Offline demo fixed.** CLIP shipped as 4 chunks + SHA256-checked reassembly + `demo.sh` preflight. Verified with sockets blocked: 8.3 s run, zero network calls. |
| 8 Sep | Claude | **Public dataset cannot supply B07 ground truth** — see *Data semantics* note below. Used for detector validation and hard negatives only. |
| 8 Sep | Claude | **Full audit at `26370d8`.** Verified by running, not reading: 92 tests pass; detector works on real CCTV (57 + 52 detections, correct classes); full pipeline runs end-to-end on real video at ~7.4 fps warm. Found: offline demo broken (338 MB runtime download), 8/12 behaviours are stubs, event graph absent, ablations unwired, 0 incidents on real footage (placeholder zones). **Closed the misdiagnosed prompt blocker** — prompts were never the problem. Queue rewritten; recording brief added. |
| 7 Sep | Codex | Added saved-prediction ablation comparison CLI with manifest-relative paths, SHA-256 input fingerprints, per-behaviour metrics, micro deltas, JSON/Markdown output, and input overwrite protection. Fixed zero-IoU threshold matching unrelated/disjoint events. Added usage and experiment limitations in `docs/ABLATIONS.md`. Verification: 92 unit tests passed, including CLI subprocess tests, outside sandbox after temporary-directory permissions blocked sandbox runs. Weekly usage: 21% used, 79% remaining. Actual pipeline variant execution and real accuracy measurements remain pending. |
| 7 Sep | Codex | Added P2 evaluation harness: `EventLabel`, greedy temporal-IoU matching by video/behaviour, per-behaviour and micro TP/FP/FN, precision/recall/F1, `n_gt`, `n_pred`, CSV and SQLite incident loaders, plus `scripts/evaluate_events.py`. Verification: `python -m pytest tests\unit -q` -> 83 passed; py_compile passed; CLI smoke against synthetic ground truth returned a valid JSON report; `npm run build` still passes. |
| 7 Sep | Codex | Extracted offline assistant logic to `handleguard/assistant/templates.py` and wired `/chat` through it. Direct tests cover cited incident IDs/timestamps, exact empty-result wording, identity refusal, incident-id lookup, and loading guardrails from `configs/sop_rules.yaml`. Verification: `python -m pytest tests\unit -q` -> 81 passed; py_compile passed for assistant/API modules; `npm run build` still passes. |
| 7 Sep | Codex | Added evidence media extraction: `handleguard/incidents/media.py` writes bounded MP4 clips and JPEG thumbnails, and `pipeline.run()` attaches/persists `clip_path` and `thumb_path`. Verification: `python -m pytest tests\unit -q` -> 76 passed; py_compile passed for media/pipeline modules; `npm run build` still passes. |
| 7 Sep | Codex | Added `apps/web/` Vite/React console: risk-sorted incident queue, band filter, stats strip, incident detail, evidence/SOP panels, review controls, clip-player slot, and guarded chat panel. Added API CORS for local Vite. Verification: `npm install` -> 0 vulnerabilities; `npm run build` passed; `python -m pytest tests\unit -q` -> 75 passed; live API returned 40 seeded rows and Vite served `http://127.0.0.1:5173/`. |
| 7 Sep | Codex | Added FastAPI service in `apps/api/main.py`: `/incidents`, `/incidents/{id}`, `/incidents/{id}/review`, `/stats`, `/clips/{file_path}`, and guarded offline `/chat`. Verification: `python -m pytest tests\unit -q` -> 74 passed; py_compile passed for API/store modules. |
| 7 Sep | Codex | Added `TASK_SHEET.md` with prioritized remaining work and usage stop condition. Started Lane A by implementing `handleguard/video/reader.py` with timestamp-based sampling and unit coverage; synthetic `drop.mp4` smoke check passed at 32 frames / 4 s / 8 fps. |
| 7 Sep | Codex | Implemented `handleguard/perception/detector.py`: YOLO-World wrapper, YAML prompt mapping, role-populated `Detection` conversion, repo-local predict/cache paths, and RuntimeError-only MPS→CPU fallback. Unit tests pass; real synthetic smoke initializes but returns 0 detections. |
| 7 Sep | Codex | Added P1 synthetic vertical slice: `FeatureExtractor`, `EventDeduper`, risk scoring/explanations, SOP-backed incident builder, and `pipeline.run`. Verification: `python -m pytest tests\unit -q` -> 71 passed; py_compile passed for new modules. |
| 7 Sep | Codex | Finished code-owned P0: ByteTrack-backed `Tracker` with IoU fallback, `NullTracker`, behaviour `FrameContext`/`TrackHistory`/registry, B01/B02/B03/B07 detectors with hard-negative tests, SQLite `IncidentStore`, and deterministic fake incident seeding. |
| 7 Sep | Claude | `scripts/render_synthetic.py` + `data/synthetic/` — Newtonian drop/throw/drag/place clips, frame-exact GT. Detector logic no longer blocked on real footage. Clips gitignored (regenerate with the script). |
| 7 Sep | Claude | Lane A foundation: `types.py` (FROZEN), `config.py`, `perception/geometry.py`. Unit tests green. Committed `00f0b03`. |
| 7 Sep | Claude | `scripts/fetch_public_data.py` — reproducible CC BY 4.0 subset fetch, upstream train/test split preserved as tune/heldout. |
| 7 Sep | Claude | MPS benchmark: **36.1 fps** (28 ms/frame). Risk R2 retired. |
| 7 Sep | Claude | `plan.md` + `project.md` written. GATE 1 approved by user. |
| 6 Sep | — | Repo scaffolded, recording guide + configs written. |

---

## Done

- [x] `handleguard/evaluation/ablations.py` + `scripts/compare_ablations.py` - saved-prediction comparison reports with provenance and deltas

- [x] `handleguard/video/reader.py` — OpenCV frame iterator with inference-fps sampling and max-resolution resize
- [x] `handleguard/perception/detector.py` — YOLO-World adapter implemented and unit-tested; prompt validation remains open
- [x] `handleguard/tracking/tracker.py` — ByteTrack-backed tracker with deterministic IoU fallback plus `NullTracker`
- [x] `handleguard/behaviours/base.py` — `FrameContext`, `TrackHistory`, shared detector contract, confidence helper
- [x] `handleguard/behaviours/registry.py` + all 12 behaviour module imports/stubs
- [x] B01 drop, B02 throw, B03 drag, B07 zone violation — implemented against normalized `TrackFeatures`
- [x] `tests/fixtures/synth.py` — synthetic track/context fixtures for behaviour tests
- [x] `handleguard/db/store.py` — SQLite incident persistence with query/get/stats/counts
- [x] `scripts/seed_fake_incidents.py` — 40 clearly seeded placeholder incidents for product work
- [x] `handleguard/features/compute.py` — scale-normalized track feature extractor
- [x] `handleguard/events/dedup.py` — one continuous drop collapses to one event
- [x] `handleguard/risk/scorer.py` + `handleguard/risk/explain.py` — contextual risk and guarded explanation text
- [x] `handleguard/incidents/builder.py` + `handleguard/incidents/sop.py` — SOP-backed incident assembly
- [x] `handleguard/incidents/media.py` — evidence MP4 and JPEG thumbnail extraction
- [x] `handleguard/pipeline.py` — deterministic synthetic clip to incident in SQLite path
- [x] `apps/api/main.py` — FastAPI incident list/detail/stats/review/clip/chat endpoints
- [x] `handleguard/assistant/templates.py` — offline assistant citations, empty-result handling, and identity refusal
- [x] `apps/web/` — Vite/React incident queue, detail, review, clip slot, and chat UI
- [x] `handleguard/evaluation/metrics.py` + `scripts/evaluate_events.py` — temporal-IoU event evaluation with counts and F1
- [x] `plan.md`, `project.md` — GATE 1 signed off
- [x] Detector weights cached to `models/yolov8s-worldv2.pt` (25 MB, committed)
- [x] MPS benchmark — 36.1 fps, logged in decisions
- [x] `.gitignore` bug fixed (was excluding the weights the offline demo needs)
- [x] `handleguard/types.py` — **FROZEN**, do not edit without announcing here
- [x] `handleguard/config.py` — single YAML entry point, `set_config_dir()` for tests
- [x] `handleguard/perception/geometry.py` + 13 hand-computed tests
- [x] `tests/unit/test_import_hygiene.py` — enforces one-directional layering
- [x] `scripts/fetch_public_data.py` — public dataset subset (~1.9 GB, gitignored)
- [x] `scripts/render_synthetic.py` — synthetic physics clips (gitignored, regenerable)

---

## Decisions log

| Decision | Rationale |
|---|---|
| **Measured 7 Sep: YOLO-World `yolov8s-worldv2` on MPS = 28 ms/frame (36.1 fps)** at 720p, imgsz=640, 5 classes | Benchmarked on the actual M3/8GB machine with a synthetic frame. Retires risk R2 — no need to drop resolution or pre-render. ~4.5x headroom over the 8 inference-fps target. Re-measure on real footage once S1 exists. |
| YOLO-World open-vocab detector | No annotation or training time available. Warehouse classes from text prompts. Cost: no custom-class mAP — report event-level metrics instead. |
| Detector weights committed to `models/` (25 MB) | Offline demo requirement. `.gitignore` corrected — it originally excluded `models/*.pt`, which would have broken a clean clone. |
| Thresholds normalized by object height, not pixels | Pixel thresholds break the moment the camera moves. See `configs/behaviours.yaml`. |
| S3 session held out, never tuned on | Only way any reported metric survives scrutiny. |
| SQLite, not Postgres | Prototype. One less service in the demo. |

---

## Claims ledger

Every number that reaches the README or deck gets a row. No row, no claim.

Hardware for every row below: **Apple M3, 8 GB, MPS**. Input: 1920×1080 CCTV
downscaled to 1280×720, `imgsz=640`, `inference_fps=8`.

| Claim | Measured? | Where measured | Value |
|---|---|---|---|
| Detection latency, p50 | ✅ 10 Sep | `artifacts/evaluation/latency.json`, 80 frames of `0_tr1.mp4`, inference_fps=24 | **28.2 ms** |
| Detection latency, p95 | ✅ 10 Sep | same | **44.8 ms** |
| End-to-end throughput | ✅ 10 Sep | same | **21.1 fps** |
| Tracking latency, p50 | ✅ 8 Sep | same | 0.5 ms |
| Feature + behaviour latency, p50 | ✅ 8 Sep | same | ≈0.1 ms combined |
| Detector on real industrial CCTV | ✅ 8 Sep | `7_tr1.mp4`, `4_te4.mp4`, one frame each | **57 and 52 detections**, classes person / cardboard box / hand trolley |
| Offline operation | ✅ 8 Sep | pipeline run with `socket.connect` patched to raise | completes in 8.3 s, **zero outbound connections** |
| Test suite | ✅ 9 Sep | `pytest -q` | **140 passing** |
| Clean-clone install | ✅ 8 Sep | fresh `git clone` to a temp dir, then `setup_offline.py` + `demo.sh --check` | **works — all assets present, CLIP reassembled, preflight OK** |
| Clean clone runs offline | ✅ 8 Sep | same clone, pipeline with `socket.connect` patched to raise | **ran in 7.4 s, zero network calls** |
| Repo size (clean clone) | ✅ 8 Sep | `du -sh` | 700 MB (detector 25 MB + CLIP chunks 338 MB + console 216 KB) |
| Demo console | ✅ 8 Sep | browser at `/` | renders; filters, separate risk/confidence columns, review controls, assistant panel; **12 behaviours shown** |
| Behaviours implemented | ✅ 8 Sep | `registry.build_all()` | **12 of 12, zero stubs** |
| API surface | ✅ 9 Sep | live curl against all routes | 7/7 respond; traversal blocked (404), bad review status rejected (422) |
| Console | ✅ 9 Sep | browser at `/` | renders, filters, sort, review persists, assistant guardrail holds, **zero console errors** |
| Evidence privacy | ✅ 9 Sep | `tests/unit/test_privacy.py` | head region blurred **in the written clip**; body preserved |
| **Reasoning-layer precision / recall / F1** | ✅ 10 Sep | `artifacts/evaluation/reasoning_eval.md`, 5 positives + 4 hard negatives, exact perception injected | **0.750 / 0.600 / 0.667** — measures reasoning only, and **thresholds were tuned on this set, so it is NOT held out** |
| **Ablation: no_tracking** | ✅ 10 Sep | same | **F1 0.667 → 0.000.** Without persistent identity nothing fires at all |
| Ablation: no_smoothing, no_event_graph | ✅ 10 Sep | same | **no delta.** Reported, not hidden — see the report for why neither is a win |
| Per-behaviour P/R on **real** footage | ❌ **NOT MEASURED** | — | **Blocked on footage. Do not quote a number.** |
| Ablation deltas on **real** video | ❌ **NOT MEASURED** | harness ready | Blocked on footage |

**Caveats that must travel with these numbers:**

- The detect *mean* is 46.9 ms, skewed by a 1491 ms first-frame model warmup.
  **Quote p50, not mean**, and say warmup is excluded.
- Detection is ~99% of pipeline time. Tracking, features and behaviour reasoning
  are together under 1 ms — the temporal layer is effectively free, which is a
  genuinely good result and worth stating.
- 21.1 fps is offline batch throughput, **not** a real-time claim.
