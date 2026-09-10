"""Event-level evaluation with temporal IoU matching."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from handleguard.db import IncidentStore
from handleguard.types import Incident


@dataclass(frozen=True, order=True)
class EventLabel:
    video: str
    behaviour: str
    start_t: float
    end_t: float
    source_id: str = ""

    @property
    def duration(self) -> float:
        return max(self.end_t - self.start_t, 0.0)


@dataclass(frozen=True)
class BehaviourMetrics:
    behaviour: str
    true_positive: int
    false_positive: int
    false_negative: int
    n_gt: int
    n_pred: int
    precision: float
    recall: float
    f1: float
    mean_temporal_iou: float


@dataclass(frozen=True)
class EvaluationReport:
    iou_threshold: float
    by_behaviour: dict[str, BehaviourMetrics]
    micro: BehaviourMetrics

    def to_dict(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "by_behaviour": {name: asdict(metrics) for name, metrics in self.by_behaviour.items()},
            "micro": asdict(self.micro),
        }


def temporal_iou(left: EventLabel, right: EventLabel) -> float:
    if left.video != right.video or left.behaviour != right.behaviour:
        return 0.0
    inter = max(min(left.end_t, right.end_t) - max(left.start_t, right.start_t), 0.0)
    union = max(left.end_t, right.end_t) - min(left.start_t, right.start_t)
    return inter / union if union > 0 else 0.0


def evaluate_events(
    ground_truth: Iterable[EventLabel],
    predictions: Iterable[EventLabel],
    *,
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    gt = sorted(ground_truth)
    pred = sorted(predictions)
    behaviours = sorted({row.behaviour for row in gt} | {row.behaviour for row in pred})
    by_behaviour = {
        behaviour: _metrics_for_behaviour(
            behaviour,
            [row for row in gt if row.behaviour == behaviour],
            [row for row in pred if row.behaviour == behaviour],
            iou_threshold=iou_threshold,
        )
        for behaviour in behaviours
    }
    micro = _micro_metrics(by_behaviour.values())
    return EvaluationReport(iou_threshold=iou_threshold, by_behaviour=by_behaviour, micro=micro)


def labels_from_csv(path: str | Path) -> list[EventLabel]:
    """Load event labels, skipping rows that assert *no* event.

    A ground-truth file lists hard negatives too — "gentle_place.mp4 contains
    nothing" is a claim worth recording, and it is what false positives are
    measured against. Those rows carry an empty behaviour and empty times, and
    contribute no EventLabel: anything predicted on that clip has nothing to
    match and is correctly counted as a false positive.
    """
    out: list[EventLabel] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            behaviour = str(row.get("behaviour") or "").strip()
            start, end = str(row.get("t_start") or "").strip(), str(row.get("t_end") or "").strip()
            if not behaviour or not start or not end:
                continue
            out.append(
                EventLabel(
                    video=str(row["video"]),
                    behaviour=behaviour,
                    start_t=float(start),
                    end_t=float(end),
                    source_id=str(row.get("id") or row.get("note") or ""),
                )
            )
    return out


def labels_from_incident_db(path: str | Path, *, limit: int = 10000) -> list[EventLabel]:
    store = IncidentStore(path)
    return incidents_to_labels(store.query(limit=limit))


def incidents_to_labels(incidents: Iterable[Incident]) -> list[EventLabel]:
    return [
        EventLabel(
            video=incident.video_id,
            behaviour=incident.name,
            start_t=float(incident.start_t),
            end_t=float(incident.end_t),
            source_id=incident.id,
        )
        for incident in incidents
    ]


def _metrics_for_behaviour(
    behaviour: str,
    gt: list[EventLabel],
    pred: list[EventLabel],
    *,
    iou_threshold: float,
) -> BehaviourMetrics:
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_row in enumerate(gt):
        for pred_idx, pred_row in enumerate(pred):
            score = temporal_iou(gt_row, pred_row)
            if score > 0.0 and score >= iou_threshold:
                candidates.append((score, gt_idx, pred_idx))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    ious: list[float] = []
    for score, gt_idx, pred_idx in sorted(candidates, reverse=True):
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        ious.append(score)

    tp = len(ious)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return BehaviourMetrics(
        behaviour=behaviour,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        n_gt=len(gt),
        n_pred=len(pred),
        precision=_safe_div(tp, tp + fp),
        recall=_safe_div(tp, tp + fn),
        f1=_f1(tp, fp, fn),
        mean_temporal_iou=round(sum(ious) / len(ious), 4) if ious else 0.0,
    )


def _micro_metrics(rows: Iterable[BehaviourMetrics]) -> BehaviourMetrics:
    values = list(rows)
    tp = sum(row.true_positive for row in values)
    fp = sum(row.false_positive for row in values)
    fn = sum(row.false_negative for row in values)
    matched = sum(row.mean_temporal_iou * row.true_positive for row in values)
    return BehaviourMetrics(
        behaviour="micro",
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        n_gt=sum(row.n_gt for row in values),
        n_pred=sum(row.n_pred for row in values),
        precision=_safe_div(tp, tp + fp),
        recall=_safe_div(tp, tp + fn),
        f1=_f1(tp, fp, fn),
        mean_temporal_iou=round(matched / tp, 4) if tp else 0.0,
    )


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def _safe_div(num: float, denom: float) -> float:
    return round(num / denom, 4) if denom else 0.0
