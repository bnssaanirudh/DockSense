"""Lightweight tracker interface used by the pipeline.

The production lane can replace the association internals with Ultralytics
ByteTrack without changing callers. For now this implements deterministic
IoU-based identity persistence plus ``NullTracker`` for the no-tracking
ablation row.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np

from handleguard.perception.geometry import iou
from handleguard.types import Detection, Frame, Track


class Tracker:
    """Persist track ids across frames.

    Uses Ultralytics ByteTrack when its optional dependencies are importable.
    Falls back to deterministic greedy IoU association for offline/test
    environments that do not have the full tracker stack installed.
    """

    def __init__(
        self,
        *,
        min_iou: float = 0.25,
        max_missed: int = 8,
        frame_rate: int = 30,
        backend: str = "auto",
    ) -> None:
        self.min_iou = min_iou
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, Track] = {}
        self._class_names: list[str] = []
        self._backend_name = "iou"
        self._byte_tracker = None
        if backend not in {"auto", "byte", "iou"}:
            raise ValueError("backend must be 'auto', 'byte', or 'iou'")
        if backend in {"auto", "byte"}:
            try:
                from ultralytics.trackers.byte_tracker import BYTETracker

                args = Namespace(
                    track_high_thresh=0.25,
                    track_low_thresh=0.1,
                    new_track_thresh=0.25,
                    track_buffer=max_missed,
                    # 0.95 accepts an association down to ~0.05 IoU. Fast
                    # vertical motion legitimately produces barely-overlapping
                    # boxes between frames; the stricter default drops the track
                    # exactly when the interesting thing is happening.
                    match_thresh=0.95,
                    fuse_score=True,
                )
                self._byte_tracker = BYTETracker(args, frame_rate=frame_rate)
                self._backend_name = "byte"
            except Exception:
                if backend == "byte":
                    raise

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()
        if self._byte_tracker is not None:
            self.__init__(
                min_iou=self.min_iou,
                max_missed=self.max_missed,
                backend=self._backend_name,
            )

    def update(self, dets: list[Detection], frame: Frame) -> list[Track]:
        if self._byte_tracker is not None:
            return self._update_byte(dets, frame)
        return self._update_iou(dets, frame)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _update_iou(self, dets: list[Detection], frame: Frame) -> list[Track]:
        unmatched_track_ids = set(self._tracks)
        matches: list[tuple[int, int]] = []

        candidates: list[tuple[float, int, int]] = []
        for det_idx, det in enumerate(dets):
            for track_id, track in self._tracks.items():
                if det.role != track.role:
                    continue
                score = iou(det.xyxy, track.xyxy)
                if score >= self.min_iou:
                    candidates.append((score, det_idx, track_id))

        used_dets: set[int] = set()
        for _, det_idx, track_id in sorted(candidates, reverse=True):
            if det_idx in used_dets or track_id not in unmatched_track_ids:
                continue
            matches.append((det_idx, track_id))
            used_dets.add(det_idx)
            unmatched_track_ids.remove(track_id)

        for det_idx, track_id in matches:
            det = dets[det_idx]
            track = self._tracks[track_id]
            track.cls = det.cls
            track.role = det.role
            track.xyxy = det.xyxy
            track.conf = det.conf
            track.last_frame = frame.index
            track.age += 1
            track.missed = 0

        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.missed += 1

        for det_idx, det in enumerate(dets):
            if det_idx in used_dets:
                continue
            track = Track(
                id=self._next_id,
                cls=det.cls,
                role=det.role,
                xyxy=det.xyxy,
                conf=det.conf,
                first_frame=frame.index,
                last_frame=frame.index,
                age=1,
            )
            self._tracks[track.id] = track
            self._next_id += 1

        stale = [
            track_id for track_id, track in self._tracks.items()
            if track.missed > self.max_missed
        ]
        for track_id in stale:
            del self._tracks[track_id]

        return [
            track for track in sorted(self._tracks.values(), key=lambda tr: tr.id)
            if track.missed == 0
        ]

    def _update_byte(self, dets: list[Detection], frame: Frame) -> list[Track]:
        results = _ByteTrackResults.from_detections(dets, self._class_names)
        tracked = self._byte_tracker.update(results, frame.image)
        active: dict[int, Track] = {}
        for row in tracked:
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            track_id = int(row[4])
            conf = float(row[5])
            class_idx = int(row[6])
            cls = self._class_names[class_idx] if 0 <= class_idx < len(self._class_names) else "unknown"
            role = results.role_for_class_idx(class_idx)
            previous = self._tracks.get(track_id)
            active[track_id] = Track(
                id=track_id,
                cls=cls,
                role=role,
                xyxy=(x1, y1, x2, y2),
                conf=conf,
                first_frame=previous.first_frame if previous else frame.index,
                last_frame=frame.index,
                age=(previous.age + 1) if previous else 1,
                missed=0,
            )
        self._tracks = active
        return [active[track_id] for track_id in sorted(active)]


class NullTracker:
    """Ablation tracker: every detection gets a fresh id."""

    def __init__(self) -> None:
        self._next_id = 1

    def reset(self) -> None:
        self._next_id = 1

    def update(self, dets: list[Detection], frame: Frame) -> list[Track]:
        tracks: list[Track] = []
        for det in dets:
            tracks.append(
                Track(
                    id=self._next_id,
                    cls=det.cls,
                    role=det.role,
                    xyxy=det.xyxy,
                    conf=det.conf,
                    first_frame=frame.index,
                    last_frame=frame.index,
                    age=1,
                )
            )
            self._next_id += 1
        return tracks


class _ByteTrackResults:
    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        roles_by_idx: dict[int, str],
    ) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self._roles_by_idx = roles_by_idx

    @classmethod
    def from_detections(cls, dets: list[Detection], class_names: list[str]):
        xyxy = np.array([det.xyxy for det in dets], dtype=np.float32).reshape((-1, 4))
        conf = np.array([det.conf for det in dets], dtype=np.float32)
        class_idx: list[int] = []
        roles_by_idx: dict[int, str] = {}
        for det in dets:
            if det.cls not in class_names:
                class_names.append(det.cls)
            idx = class_names.index(det.cls)
            class_idx.append(idx)
            roles_by_idx[idx] = det.role
        return cls(xyxy, conf, np.array(class_idx, dtype=np.float32), roles_by_idx)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, item):
        return _ByteTrackResults(
            self.xyxy[item],
            self.conf[item],
            self.cls[item],
            self._roles_by_idx,
        )

    @property
    def xywh(self) -> np.ndarray:
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
        return np.stack((x1, y1, x2 - x1, y2 - y1), axis=1).astype(np.float32)

    def role_for_class_idx(self, class_idx: int) -> str:
        return self._roles_by_idx.get(class_idx, "unknown")
