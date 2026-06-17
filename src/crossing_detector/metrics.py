"""Binary classification metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BinaryMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_binary_metrics(logits: Tensor, labels: Tensor, *, loss: float) -> BinaryMetrics:
    if logits.ndim == 2:
        if logits.shape[1] != 2:
            raise ValueError(f"Expected two-class logits, got shape {tuple(logits.shape)}.")
        predictions = torch.argmax(logits, dim=1)
    else:
        predictions = (torch.sigmoid(logits) >= 0.5).long()
    truth = labels.long()

    tp = int(((predictions == 1) & (truth == 1)).sum().item())
    fp = int(((predictions == 1) & (truth == 0)).sum().item())
    tn = int(((predictions == 0) & (truth == 0)).sum().item())
    fn = int(((predictions == 0) & (truth == 1)).sum().item())
    total = max(tp + fp + tn + fn, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / total

    return BinaryMetrics(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def format_metrics(metrics: BinaryMetrics) -> str:
    return (
        f"loss={metrics.loss:.4f} "
        f"acc={metrics.accuracy:.3f} "
        f"precision={metrics.precision:.3f} "
        f"recall={metrics.recall:.3f} "
        f"f1={metrics.f1:.3f}"
    )
