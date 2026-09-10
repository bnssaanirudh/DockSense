from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from handleguard.db.store import IncidentStore
from handleguard.pipeline import run
from handleguard.tracking.tracker import Tracker
from handleguard.types import Detection, Frame


class FallingBoxDetector:
    def __call__(self, frame: Frame) -> list[Detection]:
        y = min(100.0 + frame.index * 28.0, 520.0)
        return [Detection(cls="carton", role="product", conf=0.92, xyxy=(560.0, y, 640.0, y + 80.0))]


def test_pipeline_writes_synthetic_drop_incident_to_store(tmp_path):
    # tmp_path, not a fixed location: the store appends, so a shared path made
    # this pass on a clean checkout and fail on the next run with a stale row.
    video_path = tmp_path / "pipeline_drop.mp4"
    db_path = tmp_path / "pipeline_incidents.db"
    clip_dir = tmp_path / "clips"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    _write_blank_video(video_path)

    store = IncidentStore(db_path)
    incidents = run(
        video_path,
        detector=FallingBoxDetector(),
        tracker=Tracker(backend="iou", min_iou=0.0),
        store=store,
        video_id="pipeline-drop",
        clip_dir=clip_dir,
        max_frames=24,
    )

    assert len(incidents) == 1
    assert incidents[0].behaviour_id == "B01"
    assert incidents[0].risk.score > 0
    assert incidents[0].clip_path is not None
    assert incidents[0].thumb_path is not None
    assert (clip_dir / incidents[0].clip_path).is_file()
    assert (clip_dir / incidents[0].thumb_path).is_file()
    assert store.stats()["total"] == 1
    assert store.get(incidents[0].id).explanation == incidents[0].explanation
    assert store.get(incidents[0].id).clip_path == incidents[0].clip_path


def _write_blank_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (1280, 720),
    )
    assert writer.isOpened()
    try:
        for _ in range(32):
            writer.write(np.zeros((720, 1280, 3), dtype=np.uint8))
    finally:
        writer.release()
