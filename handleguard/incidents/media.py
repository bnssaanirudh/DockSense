"""Evidence clip and thumbnail extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2

from handleguard import config
from handleguard.privacy import blur_person_regions
from handleguard.types import BehaviourEvent


@dataclass(frozen=True)
class EvidenceAssets:
    clip_path: str | None
    thumb_path: str | None


def write_evidence_assets(
    video_path: str | Path,
    event: BehaviourEvent,
    *,
    output_dir: str | Path,
    incident_id: str,
    pre_seconds: float = 3.0,
    post_seconds: float = 4.0,
    person_boxes: list[tuple[float, list[tuple[float, float, float, float]]]] | None = None,
    blur_faces: bool | None = None,
) -> EvidenceAssets:
    """Write a bounded event clip and thumbnail.

    Returned paths are relative filenames intended for API clip serving.

    ``person_boxes`` is ``(timestamp, normalized boxes)`` pairs. Normalized
    because tracks come from downscaled inference frames while this function
    re-reads the original video; pixel boxes would land in the wrong place.
    When ``blur_faces`` is on, the head region of each person is blurred before
    the frame is stored.
    This reduces identifiability in retained evidence; it is not anonymisation,
    and nothing here should be described as such.
    """
    if blur_faces is None:
        blur_faces = bool(
            (config.behaviours().get("privacy", {}) or {}).get("blur_faces", False)
        )
    src = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(incident_id)
    clip_name = f"{stem}.mp4"
    thumb_name = f"{stem}.jpg"
    clip_path = out_dir / clip_name
    thumb_path = out_dir / thumb_name

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        return EvidenceAssets(None, None)

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 8.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total_frames <= 0 or width <= 0 or height <= 0:
            return EvidenceAssets(None, None)

        start_t = max(event.start_t - pre_seconds, 0.0)
        end_t = min(event.end_t + post_seconds, max((total_frames - 1) / fps, 0.0))
        start_frame = max(int(start_t * fps), 0)
        end_frame = min(max(int(end_t * fps), start_frame), total_frames - 1)

        # H.264 first, MPEG-4 Part 2 only as a fallback. An evidence clip that a
        # browser will not decode is not evidence: the console embeds these in a
        # <video> tag, and Chrome plays avc1 but shows a black rectangle for
        # mp4v. avc1 is also roughly 3x smaller for the same footage.
        writer = None
        for fourcc in ("avc1", "mp4v"):
            candidate = cv2.VideoWriter(
                str(clip_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is None:
            return EvidenceAssets(None, None)

        thumb_written = False
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if blur_faces and person_boxes:
                norm = _boxes_at(person_boxes, frame_index / fps if fps else 0.0)
                if norm:
                    fh_px, fw_px = frame.shape[:2]
                    frame = blur_person_regions(
                        frame,
                        [(b[0] * fw_px, b[1] * fh_px, b[2] * fw_px, b[3] * fh_px) for b in norm],
                    )
            writer.write(frame)
            if not thumb_written and frame_index >= int(event.start_t * fps):
                cv2.imwrite(str(thumb_path), frame)
                thumb_written = True
        writer.release()

        if not thumb_written:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(thumb_path), frame)

        return EvidenceAssets(
            clip_name if clip_path.exists() and clip_path.stat().st_size > 0 else None,
            thumb_name if thumb_path.exists() and thumb_path.stat().st_size > 0 else None,
        )
    finally:
        cap.release()


def _boxes_at(
    person_boxes: list[tuple[float, list[tuple[float, float, float, float]]]],
    t: float,
    *,
    tolerance: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    """Nearest recorded person boxes to time ``t``.

    Inference runs at a lower fps than the source video, so most stored frames
    fall between samples. Reusing the nearest sample within ``tolerance`` keeps
    heads covered across the gap; beyond that we blur nothing rather than
    smearing a stale box over the wrong part of the frame.
    """
    if not person_boxes:
        return []
    best_t, best = min(person_boxes, key=lambda pair: abs(pair[0] - t))
    return best if abs(best_t - t) <= tolerance else []


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "incident"
