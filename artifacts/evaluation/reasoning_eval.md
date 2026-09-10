# Reasoning-layer evaluation

```bash
python scripts/render_synthetic.py                      # both splits
python scripts/eval_reasoning.py --split tune --ablate
python scripts/eval_reasoning.py --split heldout --ablate
```

Machine-readable: `reasoning_eval_tune.json`, `reasoning_eval_heldout.json`.

## What this measures

Two halves fail for different reasons and are measured separately:

- **Perception** — YOLO-World finding boxes. Validated on real CCTV.
- **Reasoning** — tracking, temporal features, twelve detectors, dedup, risk.
  Validated here.

The harness replays the renderer's own per-frame geometry into the pipeline, so
**perception error is zero by construction**. Everything below describes the
reasoning layer alone.

## The two splits, and why there are two

The first version of this evaluation had one set of nine clips, and thresholds
and detector rules were changed in response to failures on those exact clips. It
scored **F1 1.000**, which is not a result — it is a measurement of how hard the
thresholds had been pushed towards nine specific videos.

So the renderer now samples its parameters instead of hard-coding them:

| Split | Clips | What it is |
|---|---:|---|
| `tune` | 9 | One fixed draw. The clips development happened against. |
| `heldout` | 27 | Three fresh draws per scenario — different release heights, launch speeds, lowering speeds, drag directions, stack offsets, and **carton sizes from 60 to 100 px**. Scored once. Never tuned against. |

Varying carton size is the pointed one: every threshold in `behaviours.yaml` is
in object-heights, so a 1.6× size change should be invisible. If any threshold is
secretly in pixels, this is what exposes it.

**What the held-out split controls for:** fitting thresholds to specific clips.

**What it does not control for:** bias in the generator. Both splits come from
the same renderer, the same physics, the same flat-rectangle look. A systematic
error in how this file models handling appears identically in both. Only real
footage closes that gap.

## Results (10 Sep 2026)

| Variant | tune (n=9) | **heldout (n=27)** |
|---|---:|---:|
| baseline | 1.000 | **0.933** |
| `no_tracking` | 0.000 | **0.000** |
| `no_smoothing` | 1.000 | **0.867** |
| `no_event_graph` | 1.000 | 0.933 |

Held-out baseline: precision **0.933**, recall **0.933**, over 15 labelled events
and 12 hard negatives.

**Scored twice, and that matters.** The first run read 0.875. The code then
changed in response to *real* session-1 footage — not to this split — and the
second run reads 0.933. Both are stated. A set scored twice is not strictly held
out any more, and the next genuinely clean number will have to come from footage.

**Quote the 0.933, not the 1.000.** The gap between them is the honest
measure of how much the tune number was inflated by having been tuned on.

Per behaviour, held out:

| Behaviour | TP | FP | FN | n | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| drop | 3 | 0 | 0 | 3 | 1.000 | 1.000 |
| improper_stack | 3 | 0 | 0 | 3 | 1.000 | 1.000 |
| unstable_stack | 3 | 0 | 0 | 3 | 1.000 | 1.000 |
| drag | 2 | 0 | 1 | 3 | 1.000 | 0.667 |
| throw | 3 | 3 | 0 | 3 | 0.500 | 1.000 |

## Ablations

**`no_tracking` → 0.000 on both splits.** Strip persistent identity and nothing
fires at all, because every behaviour here is defined over a sequence. This is
the row that is hard to game and the direct evidence for the central claim.

**`no_smoothing` → 0.933 → 0.867, but only on the held-out split.** On the tune
split it showed no delta at all, and the earlier report said so honestly while
noting the clips carried no jitter for smoothing to remove. Varying carton size
and speed introduced exactly that jitter, and the smoothing window now earns its
place. A mechanism whose value only appears once the inputs vary is worth knowing
about — and it is a small warning about how much the fixed tune split was hiding.

**`no_event_graph` → no delta.** Expected: the graph feeds the *risk score*, not
event detection, so event F1 is the wrong instrument. Reported rather than
dropped.

## Known failure modes (held out, not fixed)

These were left alone deliberately. Tuning them away after seeing them would turn
the held-out split into a second tuning set and destroy the only defensible
number in this document.

1. **`throw` precision 0.500 — a track ID switch mid-flight.** On the two fastest
   throws the tracker lost the box at the velocity discontinuity of release and
   re-identified it a few frames later, producing two throw events on two track
   ids. Dedup cannot merge them because dedup keys on track id. This is the same
   class of defect as the original B01 failure, now at high horizontal speed.
2. **One carried box read as a throw.** B02 infers "unsupported" from the absence
   of person-box overlap. At a high carry height the person box stops overlapping
   the carton, so a carry at speed satisfies the throw condition. The proxy is
   the weakness, not the threshold.
3. **`drag` recall 0.667 — one slow drag missed.** The slowest sampled drag on the
   largest sampled carton travels under `min_distance_heights` within the
   detector's window. Whether 1.5 object-heights is the right bar is a question
   for real footage, not for this generator.

## Next

All of it is blocked behind the same thing: **real recorded footage**, held out,
scored once. Brief in `STATE.md`.
