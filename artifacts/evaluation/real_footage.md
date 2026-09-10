# Real footage — session 1

```bash
python scripts/prepare_footage.py --session session1   # crop to the camera pane
python scripts/eval_real.py --run                      # run pipeline, cache, score
```

Machine-readable: `real_footage.json`, `real_footage_incidents.json`.

## What the footage is

Seven clips, ~3 minutes total, of genuine warehouse loading-bay operations:
rolling and dropping cartons, throwing seating cartons and mattresses, dragging
packets and a cupboard, stepping on cartons, heavy product placed on light.

It is **not direct camera output**. Every clip is a phone recording of an NVMS
playback window, so each frame carries software chrome — menu bar, timeline
scrubber, calendar sidebar — plus a burned-in caption card, around a CCTV pane
that is itself already a lossy encode. `configs/rois.yaml` crops each clip to its
camera pane before anything else runs. Some clips also carry red circles and
arrows the recorder drew *inside* the pane to point at the behaviour; cropping
cannot remove those and they remain in the evidence clips.

The recording brief in `STATE.md` asked for a fixed tripod, taped zones, marked
boxes and a `takes.csv` written at record time. None of that applies here: this
is opportunistic footage of real operations, which is more representative than a
staged session and much harder to score.

## What this measures, and what it cannot

**Clip-level behaviour presence.** For each clip, did the system report the
behaviour the recorder said was there?

It cannot measure:

- **Precision.** There are **no hard-negative clips** — no footage of ordinary
  handling that must stay silent. Every clip is a positive, so a detector that
  fired constantly would score perfectly. Alarms per minute is reported instead,
  and it is not precision.
- **Timing.** No start/end times were logged at record time, and inventing spans
  afterwards would mean scoring the system against labels drawn after seeing what
  the system did.

Split: 3 clips tuned on, 4 held out. Named in `data/raw/session1/takes.csv`.

## Result

**1 of 9 expected behaviours reported.** That is the honest number and it is a
poor one. It is also the most useful number in this repository, because every
synthetic result above it was measured on footage the system could not fail on
in these particular ways.

The system is not silent — it produced incidents on 5 of 7 clips, at 2 to 24
alarms per minute — but they are mostly `rough_handling` and
`manual_heavy_handling`, the two behaviours with the loosest conditions, rather
than the specific behaviour each clip was recorded to show.

## What real footage exposed that rendered clips could not

Four defects, three of them fixed on the tune clips only:

1. **`min_track_age_frames` was declared in `behaviours.yaml` and read by
   nothing.** A track one frame old has no velocity history, so its first real
   measurement is a full-magnitude jump — on soft footage that jump is detector
   jitter, and it was firing `throw` on clips of people dragging. Now enforced in
   `FeatureExtractor`. **Fixed.**
2. **A single horizontal floor line cannot describe a perspective view.** One line
   puts a box on the floor at one depth and a metre in the air at another, which
   is why B03 never fired on any of the four dragging clips. The nearest person's
   feet are now used as a local, calibration-free ground reference. **Fixed.**
3. **`held_by_person` was true essentially always.** The test was horizontal
   overlap plus any vertical intersection, which in a loading bay means "somebody
   is standing behind it". Measured: a carton being thrown across a gap read as
   held in **100%** of frames, so B02 could not fire on a real throw by
   construction. Now requires a real share of the product's area to be inside the
   person's box. **Fixed — and B02 still does not fire on the real throw clips.**
4. **Throw and drag are not reliably separable on this footage.** The physics that
   distinguishes them — a ballistic arc — is not observable here. Vertical
   acceleration on the tune clips has a p99 of ~48 object-heights/s² against a
   free-fall value of ~20: the signal is entirely below the noise floor of the
   detector's box jitter at this resolution. A ballistic test was designed and
   then **not built**, because measuring the feature first showed it could not
   work. **Not fixed, and not fixable by tuning.**

## What would actually move this

In rough order of value per hour:

1. **One clip of ordinary handling where nothing bad happens.** Sixty seconds.
   Without it, precision on real footage stays permanently unmeasurable, and the
   alarm rates above cannot be interpreted at all.
2. **Start/end times logged at record time**, per `takes.csv`. Turns this from a
   presence check into a real precision/recall measurement.
3. **Direct camera export instead of a phone pointed at a monitor.** Removes two
   lossy encodes, the moiré, the chrome and the drawn-on annotations in one step.
4. Higher-resolution or closer framing, so a carton is more than ~200 px tall and
   the acceleration signal rises above box jitter.

## Related

Reasoning-layer results on synthetic clips, where perception is held perfect:
`reasoning_eval.md`. Those numbers (held-out F1 0.875) describe the reasoning
logic in isolation. This document is what happens when that logic meets a real
camera, and the gap between the two is the honest measure of how much the
synthetic result is worth.
