import torch

from crossing_detector.metrics import compute_binary_metrics


def test_binary_metrics_counts_confusion_matrix():
    logits = torch.tensor(
        [
            [0.0, 4.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [4.0, 0.0],
        ]
    )
    labels = torch.tensor([1, 0, 0, 1])

    metrics = compute_binary_metrics(logits, labels, loss=0.5)

    assert metrics.tp == 1
    assert metrics.tn == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
