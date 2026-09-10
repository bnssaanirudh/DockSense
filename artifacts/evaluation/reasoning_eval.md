# Reasoning-layer evaluation

Regenerate with:

```bash
python scripts/render_synthetic.py
python scripts/eval_reasoning.py --ablate
```

Machine-readable results: `reasoning_eval.json`.

## What this measures

The system has two halves that fail for different reasons:

- **Perception** — YOLO-World finding boxes. Validated separately, on real CCTV.
- **Reasoning** — tracking, temporal features, twelve behaviour detectors, dedup,
  risk. Validated here.

This harness replays the renderer's own per-frame box geometry into the pipeline,
so **perception error is zero by construction**. Every number below describes the
reasoning layer alone.

Set: 5 labelled positives (drop, throw, drag, improper_stack, unstable_stack) and
4 hard negatives (carry, gentle_place, good_stack, static). Temporal IoU 0.3.

## Results (10 Sep 2026)

| Variant | Precision | Recall | F1 |
|---|---:|---:|---:|
| baseline | 1.000 | 1.000 | **1.000** |
| `no_tracking` | 0.000 | 0.000 | **0.000** |
| `no_smoothing` | 1.000 | 1.000 | 1.000 |
| `no_event_graph` | 1.000 | 1.000 | 1.000 |

Per-clip predictions:

```
drop.mp4            -> drop
throw.mp4           -> throw
drag.mp4            -> drag
improper_stack.mp4  -> improper_stack
unstable_stack.mp4  -> unstable_stack
carry.mp4           -> (none)
gentle_place.mp4    -> (none)
good_stack.mp4      -> (none)
static.mp4          -> (none)
```

## Reading these honestly

**A 1.000 is not an accuracy claim, and treating it as one would be wrong.**

The set has nine clips. Every threshold and several detector rules were changed
*in response to failures on these exact clips*, so the thing being measured and
the thing being optimised now share a generating function. That is the oracle
problem, and it is the single largest methodological weakness in this project.
The number says the reasoning logic is self-consistent on the cases it was
debugged against. It says nothing about a warehouse.

What the run is genuinely good for:

- **It found real defects that no test caught.** See below — the failures were
  informative even though the final score is not.
- **`no_tracking` collapsing to 0.000** is the one row that is hard to game. Strip
  persistent identity and nothing fires at all, because every behaviour here is
  defined over a sequence. That is direct evidence for the central design claim.

What it is not:

- Not a field accuracy figure. **Real footage of these behaviours does not exist
  yet**, and until it does no per-behaviour precision/recall belongs in a slide.
- Not a perception result. YOLO-World correctly returns nothing on flat
  rectangles; running it here would measure the renderer's photorealism.

`no_smoothing` and `no_event_graph` show no delta, reported rather than dropped.
These clips carry zero detector jitter for smoothing to remove, and the event
graph feeds the *risk score* rather than event detection, so event F1 is the
wrong instrument for it. A switch that changes nothing is worth knowing about.

## Defects this run exposed

Listed because the bugs are the actual output of the exercise.

1. **Tracked boxes were offset by half their own size.** `_ByteTrackResults.xywh`
   returned top-left-based `(x, y, w, h)`; BYTETracker feeds that straight into
   `xywh2ltwh`, which expects centre-based. Invisible to relative measures like
   fall distance — which is why B01 passed — but it silently broke every absolute
   one: floor gap, zone containment, support geometry.
2. **Horizontal distances were scaled by frame height.** `travel_heights` used the
   frame height for both axes, but `cx` is normalised by width. On 16:9 every
   horizontal travel read 0.5625× its true value, so a 1.9-object-height drag
   measured 1.06 and never crossed its 1.5 threshold.
3. **"Sustained" meant "the track has existed this long."** Five static-geometry
   detectors called `sustained_seconds(window, lambda f: True)`, which measures
   track age, not how long the *condition* held. A stack still being lowered
   satisfied it. Replaced with `SustainedCondition`, which clocks the actual
   configuration and restarts after a lapse.
4. **B05's two thresholds were mutually unsatisfiable.** Overlap was measured as a
   fraction of the *upper* box's width, so a larger `area_ratio` — the very thing
   B05 exists to catch — forced that fraction down. Now measured against the
   support below, where "covers the whole thing" reads as 1.0.
5. **Suppression was always one frame late.** `recent_events` was snapshotted once
   per frame, so a detector deferring to a more specific one still emitted a
   duplicate on the frame they both first fired. Now refreshed per detector, with
   registry order putting the specific detector first.
6. **The renderer's own labels disagreed with its physics.** The stacking clips
   started their box 300 px above the target at 80 px/s, so the stack existed for
   the last 0.25 s of a 5 s clip while the label claimed 3 s of it — and one of
   them started *below* its target and never moved at all. The label described the
   intent correctly, so the physics moved to match rather than the threshold.

## Next

Everything here is blocked behind the same thing: **real recorded footage**, held
out, scored once. The recording brief is in `STATE.md`.
