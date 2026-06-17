import torch

from crossing_detector.metrics import BinaryMetrics
from crossing_detector.train import (
    infer_qat_dense_after_stage_epoch,
    make_scheduler,
    quantized_eval_reason,
    should_start_finetuning,
    step_scheduler,
)


def metrics(train_f1: float, val_f1: float) -> tuple[BinaryMetrics, BinaryMetrics]:
    train = BinaryMetrics(0.5, train_f1, train_f1, train_f1, train_f1, 1, 0, 1, 0)
    val = BinaryMetrics(0.5, val_f1, val_f1, val_f1, val_f1, 1, 0, 1, 0)
    return train, val


def test_finetuning_starts_when_validation_f1_is_good_enough():
    train, val = metrics(0.75, 0.72)

    reason = should_start_finetuning(
        train_metrics=train,
        val_metrics=val,
        stale_epochs=0,
        completed_frozen_epochs=1,
        max_frozen_epochs=12,
        min_finetune_f1=0.70,
        max_head_train_f1=0.95,
        frozen_patience=3,
    )

    assert reason == "val_f1_reached_0.700"


def test_finetuning_starts_when_frozen_patience_is_exhausted():
    train, val = metrics(0.55, 0.45)

    reason = should_start_finetuning(
        train_metrics=train,
        val_metrics=val,
        stale_epochs=3,
        completed_frozen_epochs=4,
        max_frozen_epochs=12,
        min_finetune_f1=0.70,
        max_head_train_f1=0.95,
        frozen_patience=3,
    )

    assert reason == "frozen_patience_3"


def test_quantized_eval_only_runs_after_finetuning_starts():
    assert (
        quantized_eval_reason(
            enabled=True,
            stage="frozen",
            stage_epoch=20,
            interval=3,
            dense_after_stage_epoch=7,
        )
        is None
    )
    assert (
        quantized_eval_reason(
            enabled=True,
            stage="layer4",
            stage_epoch=1,
            interval=3,
            dense_after_stage_epoch=7,
        )
        == "first_finetune_epoch"
    )


def test_quantized_eval_switches_to_dense_near_previous_best_window():
    assert (
        quantized_eval_reason(
            enabled=True,
            stage="layer4",
            stage_epoch=6,
            interval=3,
            dense_after_stage_epoch=7,
        )
        == "interval_3"
    )
    assert (
        quantized_eval_reason(
            enabled=True,
            stage="layer4",
            stage_epoch=7,
            interval=3,
            dense_after_stage_epoch=7,
        )
        == "dense_after_stage_epoch_7"
    )
    assert (
        quantized_eval_reason(
            enabled=True,
            stage="all",
            stage_epoch=1,
            interval=3,
            dense_after_stage_epoch=7,
        )
        == "full_finetune_every_epoch"
    )


def test_qat_dense_start_falls_back_without_reference_history():
    assert infer_qat_dense_after_stage_epoch(None, fallback=7, lookback=1) == 7


def test_qat_dense_start_can_use_previous_history(tmp_path):
    history_path = tmp_path / "metrics_history.json"
    history_path.write_text(
        """
        [
          {"stage": "frozen", "stage_epoch": 1, "val": {"f1": 0.90}},
          {"stage": "layer4", "stage_epoch": 3, "val": {"f1": 0.91}},
          {"stage": "layer4", "stage_epoch": 7, "val": {"f1": 0.94}},
          {"stage": "layer4", "stage_epoch": 8, "val": {"f1": 0.92}}
        ]
        """,
        encoding="utf-8",
    )

    assert infer_qat_dense_after_stage_epoch(history_path, fallback=7, lookback=1) == 6


def test_plateau_scheduler_reduces_lr_when_validation_f1_stalls():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = make_scheduler(
        optimizer,
        scheduler_name="plateau",
        inverse_decay=0.08,
        plateau_factor=0.5,
        plateau_patience=0,
        plateau_threshold=0.001,
        min_lr=0.0,
    )
    _, val = metrics(0.5, 0.8)

    step_scheduler(scheduler, val)
    assert optimizer.param_groups[0]["lr"] == 0.1

    step_scheduler(scheduler, val)
    assert optimizer.param_groups[0]["lr"] == 0.05
