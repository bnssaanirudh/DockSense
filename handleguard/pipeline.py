"""End-to-end offline inference pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Protocol

from handleguard import config
from handleguard.behaviours.base import FrameContext, TrackHistory
from handleguard.behaviours.registry import build_all
from handleguard.events.dedup import EventDeduper
from handleguard.events.graph import build_graph
from handleguard.features.compute import FeatureExtractor
from handleguard.incidents.builder import build_incident, incident_id_for
from handleguard.incidents.media import write_evidence_assets
from handleguard.metrics.latency import Timings
from handleguard.risk.scorer import score_event
from handleguard.tracking.tracker import NullTracker, Tracker
from handleguard.types import Detection, Frame, Incident, Track
from handleguard.video.reader import iter_frames

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_DIR = ROOT / "data" / "clips"


@dataclass(frozen=True)
class PipelineFlags:
    """Ablation switches.

    The claim this project makes is that temporal reasoning beats frame-only
    rules. That claim is only worth anything if the temporal parts can actually
    be turned off and the cost measured, so each flag disables one mechanism
    while leaving everything else identical:

    - ``use_tracking``        NullTracker — a fresh id per detection per frame, so
                              no track persists and nothing temporal can accumulate.
    - ``use_smoothing``       velocity from a single frame pair instead of a
                              smoothed window, i.e. raw detector jitter.
    - ``use_zones``           no zone polygons, so zone-derived behaviours and the
                              location risk component go dark.
    - ``use_contextual_risk`` risk collapses to behaviour severity alone, dropping
                              kinematics/support/duration/frequency/zone context.
    - ``use_event_graph``     no cross-event graph, so recurrence context is lost.

    Defaults are the full system. ``from_config`` reads the ``ablation:`` block in
    behaviours.yaml so a run can be configured without touching code.
    """

    use_tracking: bool = True
    use_smoothing: bool = True
    use_event_graph: bool = True
    use_zones: bool = True
    use_contextual_risk: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> "PipelineFlags":
        section = (cfg or config.behaviours()).get("ablation", {}) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: bool(v) for k, v in section.items() if k in known})

    def label(self) -> str:
        """Short name for reports: 'baseline', or the disabled mechanisms."""
        off = [f.name for f in fields(self) if not getattr(self, f.name)]
        return "baseline" if not off else "no_" + "+".join(n.removeprefix("use_") for n in off)


class DetectorLike(Protocol):
    def __call__(self, frame: Frame) -> list[Detection]:
        ...


class TrackerLike(Protocol):
    def update(self, dets: list[Detection], frame: Frame) -> list[Track]:
        ...


def run(
    video_path: str | Path,
    *,
    detector: DetectorLike | None = None,
    tracker: TrackerLike | None = None,
    feature_extractor: FeatureExtractor | None = None,
    detectors: list[Any] | None = None,
    store: Any | None = None,
    video_id: str | None = None,
    session: str = "synthetic",
    camera: str = "demo_cam_1",
    clip_dir: str | Path | None = DEFAULT_CLIP_DIR,
    write_clips: bool = True,
    max_frames: int | None = None,
    flags: PipelineFlags | None = None,
    timings: Timings | None = None,
) -> list[Incident]:
    behaviour_cfg = config.behaviours()
    video_cfg = behaviour_cfg.get("video", {})
    inference_fps = float(video_cfg.get("inference_fps", 8))
    max_res = tuple(video_cfg.get("max_resolution", (1280, 720)))
    flags = flags or PipelineFlags()

    if detector is None:
        from handleguard.perception.detector import YoloWorldDetector

        detector = YoloWorldDetector()

    if tracker is None:
        tracker = (
            Tracker(frame_rate=int(round(inference_fps)))
            if flags.use_tracking
            else NullTracker()
        )
    if feature_extractor is None:
        feature_extractor = FeatureExtractor(
            camera=camera,
            # window_frames=1 means "this frame vs the last one" — no smoothing.
            window_frames=None if flags.use_smoothing else 1,
            zones_cfg=None if flags.use_zones else {"cameras": {}},
        )
    detectors = detectors or build_all(behaviour_cfg)

    history = TrackHistory(max_seconds=_history_seconds(behaviour_cfg))
    deduper = EventDeduper(behaviour_cfg)
    # Person boxes for privacy blurring of stored clips, as (t, normalized boxes).
    # Normalized because tracks are in inference resolution while the clip writer
    # re-reads the ORIGINAL video — storing pixels here would blur the wrong
    # region at any resolution other than the one we inferred at. Keyed by time
    # rather than frame index because the reader skips frames.
    person_boxes: list[tuple[float, list[tuple[float, float, float, float]]]] = []
    prev_t = 0.0
    video_id = video_id or Path(video_path).stem

    if hasattr(tracker, "reset"):
        tracker.reset()
    feature_extractor.reset()
    for behaviour in detectors:
        if hasattr(behaviour, "reset"):
            behaviour.reset()

    for frame in iter_frames(video_path, inference_fps=inference_fps, max_res=(int(max_res[0]), int(max_res[1]))):
        if max_frames is not None and frame.index >= max_frames:
            break
        with _stage(timings, "detect"):
            dets = detector(frame)
        with _stage(timings, "track"):
            tracks = tracker.update(dets, frame)
        with _stage(timings, "features"):
            feats = feature_extractor.update(tracks, frame)
        history.push(feats)
        actors = [
            (t.xyxy[0] / frame.w, t.xyxy[1] / frame.h, t.xyxy[2] / frame.w, t.xyxy[3] / frame.h)
            for t in tracks
            if t.role == "actor"
        ]
        if actors:
            person_boxes.append((frame.t, actors))
        dt = frame.t - prev_t if frame.index else 0.0
        prev_t = frame.t
        ctx = FrameContext(
            frame_index=frame.index,
            t=frame.t,
            dt=dt,
            fw=frame.w,
            fh=frame.h,
            tracks={track.id: track for track in tracks},
            feats=feats,
            history=history,
            zones=config.zones() if flags.use_zones else {"cameras": {}},
            cfg=behaviour_cfg,
            recent_events=deduper.settled(),
        )
        with _stage(timings, "behaviours"):
            for behaviour in detectors:
                # Refresh per detector, not per frame: suppression is useless if
                # the event it defers to only becomes visible on the next frame,
                # because that one frame still becomes a duplicate incident.
                for event in behaviour.update(replace(ctx, recent_events=deduper.settled())):
                    deduper.push(event)

    incidents = []
    evidence_cfg = behaviour_cfg.get("evidence", {})
    settled_events = deduper.flush()

    # Temporal event graph: relates the deduplicated events to one another so a
    # repeated behaviour on the same entity is scored as recurrence rather than
    # as N unrelated incidents. Disabled by the ablation flag.
    graph = build_graph(settled_events) if flags.use_event_graph else None
    recurrence = graph.recurrence() if graph else {}

    for event in settled_events:
        repeats = max(
            (recurrence.get((event.behaviour_id, tid), 1) for tid in event.track_ids),
            default=1,
        )
        if repeats > 1:
            event = replace(
                event,
                evidence={**event.evidence, "repeat_count": repeats},
            )
        with _stage(timings, "risk"):
            risk = score_event(event, contextual=flags.use_contextual_risk)
        incident_id = incident_id_for(event, video_id)
        clip_path = thumb_path = None
        if write_clips and clip_dir is not None:
            with _stage(timings, "evidence"):
                assets = write_evidence_assets(
                    video_path,
                    event,
                    output_dir=clip_dir,
                    incident_id=incident_id,
                    pre_seconds=float(evidence_cfg.get("pre_event_seconds", 3.0)),
                    post_seconds=float(evidence_cfg.get("post_event_seconds", 4.0)),
                    person_boxes=person_boxes,
                )
            clip_path = assets.clip_path
            thumb_path = assets.thumb_path
        incidents.append(
            build_incident(
                event,
                video_id=video_id,
                session=session,
                camera=camera,
                risk=risk,
                clip_path=clip_path,
                thumb_path=thumb_path,
                incident_id=incident_id,
            )
        )
    if store is not None:
        if hasattr(store, "init"):
            store.init()
        for incident in incidents:
            store.add(incident)
    return incidents


@contextmanager
def _stage(timings: Timings | None, name: str):
    """No-op when instrumentation is off, so the hot path pays nothing."""
    if timings is None:
        yield
        return
    with timings.stage(name):
        yield


def _history_seconds(cfg: dict[str, Any]) -> float:
    windows = [float(v) for v in cfg.get("dedup", {}).values()]
    windows.extend(
        float(section.get("cooldown_seconds", 0.0))
        for section in cfg.values()
        if isinstance(section, dict) and "cooldown_seconds" in section
    )
    return max(windows + [5.0])
