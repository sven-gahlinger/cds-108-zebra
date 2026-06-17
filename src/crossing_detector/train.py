"""Train a SwissImage pedestrian-crossing classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from crossing_detector.data import build_dataloaders, resolve_data_roots
from crossing_detector.metrics import BinaryMetrics, compute_binary_metrics, format_metrics
from crossing_detector.models import (
    BACKBONE_SPECS,
    SwissImageCrossingClassifier,
    build_crossing_classifier,
    set_frozen_batchnorm_eval,
)
from crossing_detector.quantize import evaluate_model, make_calibration_loader, quantize_model_for_cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Folder containing train/ and val/.")
    parser.add_argument("--train-root", type=Path, default=None, help="Explicit training ImageFolder root.")
    parser.add_argument("--val-root", type=Path, default=None, help="Explicit validation ImageFolder root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start model weights from a previous checkpoint. Optimizer state is not restored.",
    )
    parser.add_argument("--positive-class", default="crossing")
    parser.add_argument("--backbone", default="fmow-resnet50", choices=sorted(BACKBONE_SPECS))
    parser.add_argument("--image-size", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze-epochs", type=int, default=12, help="Maximum MLP-only warmup epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--full-finetune-epochs", type=int, default=0)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("plateau", "inverse", "none"), default="plateau")
    parser.add_argument(
        "--lr-aggression-decay",
        type=float,
        default=0.08,
        help="Only used with --lr-scheduler inverse.",
    )
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-patience", type=int, default=2)
    parser.add_argument("--lr-plateau-threshold", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-finetune-f1", type=float, default=0.70)
    parser.add_argument("--max-head-train-f1", type=float, default=0.95)
    parser.add_argument("--frozen-patience", type=int, default=3)
    parser.add_argument("--finetune-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--pos-weight", default="auto", help="Use 'auto', 'none', or a numeric value.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--qat-eval",
        action="store_true",
        help="Train float32, but periodically evaluate a CPU int8 shadow model after CNN fine-tuning starts.",
    )
    parser.add_argument("--qat-backend", default="auto", help="Quantized CPU backend, or 'auto'.")
    parser.add_argument("--qat-calibration-split", choices=("train", "val"), default="train")
    parser.add_argument("--qat-calibration-samples", type=int, default=128)
    parser.add_argument("--qat-eval-interval", type=int, default=3)
    parser.add_argument(
        "--qat-dense-after-stage-epoch",
        type=int,
        default=7,
        help="Run int8 validation every fine-tune epoch from this stage epoch onward.",
    )
    parser.add_argument(
        "--qat-reference-history",
        type=Path,
        default=None,
        help="Optional previous metrics_history.json used to infer the dense int8-validation window.",
    )
    parser.add_argument(
        "--qat-dense-lookback",
        type=int,
        default=1,
        help="Start dense int8 validation this many fine-tune epochs before the previous best epoch.",
    )
    return parser.parse_args()


def configure_stage(model: SwissImageCrossingClassifier, stage: str) -> None:
    if stage == "frozen":
        model.freeze_backbone()
    elif stage == "layer4":
        model.unfreeze_last_stage()
    elif stage == "all":
        model.unfreeze_backbone()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def make_optimizer(
    model: SwissImageCrossingClassifier,
    *,
    stage: str,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    groups: list[dict[str, Any]] = [{"params": head_parameters, "lr": lr_head}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def make_inverse_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    decay: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Reduce training aggressiveness as epochs accumulate."""

    if decay < 0:
        raise ValueError("--lr-aggression-decay must be zero or greater.")

    def factor(epoch_index: int) -> float:
        return 1.0 / (1.0 + decay * epoch_index)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    inverse_decay: float,
    plateau_factor: float,
    plateau_patience: int,
    plateau_threshold: float,
    min_lr: float,
):
    if scheduler_name == "none":
        return None
    if scheduler_name == "inverse":
        return make_inverse_epoch_scheduler(optimizer, decay=inverse_decay)
    if scheduler_name != "plateau":
        raise ValueError(f"Unknown LR scheduler: {scheduler_name}")
    if not 0.0 < plateau_factor < 1.0:
        raise ValueError("--lr-plateau-factor must be greater than 0 and less than 1.")
    if plateau_patience < 0:
        raise ValueError("--lr-plateau-patience must be zero or greater.")
    if plateau_threshold < 0:
        raise ValueError("--lr-plateau-threshold must be zero or greater.")
    if min_lr < 0:
        raise ValueError("--lr-min must be zero or greater.")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=plateau_factor,
        patience=plateau_patience,
        threshold=plateau_threshold,
        threshold_mode="abs",
        min_lr=min_lr,
    )


def step_scheduler(scheduler, val_metrics: BinaryMetrics) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_metrics.f1)
    else:
        scheduler.step()


def resolve_pos_weight(value: str, label_counts: dict[str, int], device: torch.device) -> Tensor | None:
    if value == "none":
        return None
    if value == "auto":
        positive = label_counts["positive"]
        negative = label_counts["negative"]
        if positive == 0:
            raise ValueError("Cannot use --pos-weight auto with zero positive training samples.")
        return torch.tensor([1.0, negative / positive], dtype=torch.float32, device=device)
    return torch.tensor([1.0, float(value)], dtype=torch.float32, device=device)


def train_one_epoch(
    model: SwissImageCrossingClassifier,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    stage: str,
) -> BinaryMetrics:
    model.train()
    if stage != "all":
        set_frozen_batchnorm_eval(model.backbone)

    total_loss = 0.0
    seen = 0
    logits_all: list[Tensor] = []
    labels_all: list[Tensor] = []
    progress = tqdm(loader, desc=f"train/{stage}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        seen += batch_size
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        progress.set_postfix(loss=total_loss / max(seen, 1))

    return compute_binary_metrics(
        torch.cat(logits_all),
        torch.cat(labels_all),
        loss=total_loss / max(seen, 1),
    )


@torch.inference_mode()
def evaluate(
    model: SwissImageCrossingClassifier,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> BinaryMetrics:
    model.eval()
    total_loss = 0.0
    seen = 0
    logits_all: list[Tensor] = []
    labels_all: list[Tensor] = []

    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        seen += batch_size
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())

    return compute_binary_metrics(
        torch.cat(logits_all),
        torch.cat(labels_all),
        loss=total_loss / max(seen, 1),
    )


def checkpoint_payload(
    model: SwissImageCrossingClassifier,
    *,
    args: argparse.Namespace,
    epoch: int,
    stage: str,
    val_metrics: BinaryMetrics,
    class_to_idx: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "stage": stage,
        "val_metrics": val_metrics.to_dict(),
        "class_to_idx": class_to_idx,
        "positive_class": args.positive_class,
        "backbone": args.backbone,
    }
    if extra is not None:
        payload.update(extra)
    return payload


def load_initial_checkpoint(
    model: SwissImageCrossingClassifier,
    checkpoint_path: Path,
    *,
    expected_backbone: str,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_backbone = payload.get("backbone") or payload.get("args", {}).get("backbone")
    if checkpoint_backbone is not None and checkpoint_backbone != expected_backbone:
        raise ValueError(
            f"Checkpoint backbone {checkpoint_backbone!r} does not match requested "
            f"backbone {expected_backbone!r}."
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_stage": payload.get("stage"),
        "checkpoint_val_metrics": payload.get("val_metrics"),
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def improved(current: float, best: float, *, min_delta: float) -> bool:
    return current > best + min_delta


def should_start_finetuning(
    *,
    train_metrics: BinaryMetrics,
    val_metrics: BinaryMetrics,
    stale_epochs: int,
    completed_frozen_epochs: int,
    max_frozen_epochs: int,
    min_finetune_f1: float,
    max_head_train_f1: float,
    frozen_patience: int,
) -> str | None:
    if val_metrics.f1 >= min_finetune_f1:
        return f"val_f1_reached_{min_finetune_f1:.3f}"
    if train_metrics.f1 >= max_head_train_f1:
        return f"head_train_f1_reached_{max_head_train_f1:.3f}"
    if frozen_patience > 0 and stale_epochs >= frozen_patience:
        return f"frozen_patience_{frozen_patience}"
    if completed_frozen_epochs >= max_frozen_epochs:
        return f"max_frozen_epochs_{max_frozen_epochs}"
    return None


def infer_qat_dense_after_stage_epoch(
    history_path: Path | None,
    *,
    fallback: int,
    lookback: int,
) -> int:
    if history_path is None or not history_path.exists():
        return max(1, fallback)

    records = json.loads(history_path.read_text(encoding="utf-8"))
    fine_tune_records = [
        record
        for record in records
        if record.get("stage") in {"layer4", "all"} and isinstance(record.get("val"), dict)
    ]
    if not fine_tune_records:
        return max(1, fallback)

    best_record = max(
        fine_tune_records,
        key=lambda record: float(record.get("val", {}).get("f1", -1.0)),
    )
    best_stage_epoch = int(best_record.get("stage_epoch", fallback))
    return max(1, best_stage_epoch - max(0, lookback))


def quantized_eval_reason(
    *,
    enabled: bool,
    stage: str,
    stage_epoch: int,
    interval: int,
    dense_after_stage_epoch: int,
) -> str | None:
    if not enabled or stage == "frozen":
        return None
    if stage == "all":
        return "full_finetune_every_epoch"
    if stage_epoch <= 0:
        return None
    if stage_epoch >= dense_after_stage_epoch:
        return f"dense_after_stage_epoch_{dense_after_stage_epoch}"
    if stage_epoch == 1:
        return "first_finetune_epoch"
    if interval > 0 and stage_epoch % interval == 0:
        return f"interval_{interval}"
    return None


def save_training_checkpoint(
    model: SwissImageCrossingClassifier,
    *,
    args: argparse.Namespace,
    epoch: int,
    stage: str,
    val_metrics: BinaryMetrics,
    class_to_idx: dict[str, int],
    output_dir: Path,
    names: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = checkpoint_payload(
        model,
        args=args,
        epoch=epoch,
        stage=stage,
        val_metrics=val_metrics,
        class_to_idx=class_to_idx,
        extra=extra,
    )
    for name in names:
        torch.save(payload, output_dir / name)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots = resolve_data_roots(
        data_root=args.data_root,
        train_root=args.train_root,
        val_root=args.val_root,
    )
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
        roots,
        positive_class=args.positive_class,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train and validation class folders differ: "
            f"{train_dataset.classes} vs {val_dataset.classes}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_crossing_classifier(
        args.backbone,
        pretrained=not args.no_pretrained and args.init_checkpoint is None,
        dropout=args.dropout,
    ).to(device)
    initial_checkpoint: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        initial_checkpoint = load_initial_checkpoint(
            model,
            args.init_checkpoint,
            expected_backbone=args.backbone,
            device=device,
        )

    pos_weight = resolve_pos_weight(args.pos_weight, train_dataset.label_counts, device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    cpu_device = torch.device("cpu")
    qat_dense_after_stage_epoch = infer_qat_dense_after_stage_epoch(
        args.qat_reference_history,
        fallback=args.qat_dense_after_stage_epoch,
        lookback=args.qat_dense_lookback,
    )
    qat_calibration_loader = None
    qat_criterion = None
    if args.qat_eval:
        qat_calibration_root = roots.train if args.qat_calibration_split == "train" else roots.val
        qat_calibration_loader = make_calibration_loader(
            qat_calibration_root,
            image_size=args.image_size,
            positive_class=args.positive_class,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        qat_pos_weight = resolve_pos_weight(args.pos_weight, train_dataset.label_counts, cpu_device)
        qat_criterion = nn.CrossEntropyLoss(weight=qat_pos_weight)

    config = {
        **vars(args),
        "data_roots": {"train": str(roots.train), "val": str(roots.val)},
        "train_counts": train_dataset.label_counts,
        "val_counts": val_dataset.label_counts,
        "device": str(device),
        "class_weight_resolved": None if pos_weight is None else pos_weight.detach().cpu().tolist(),
        "backbone_description": BACKBONE_SPECS[args.backbone].description,
        "qat_dense_after_stage_epoch_resolved": qat_dense_after_stage_epoch,
        "initial_checkpoint": initial_checkpoint,
    }
    save_json(args.output_dir / "run_config.json", config)

    history: list[dict[str, Any]] = []
    quantized_eval_history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_finetuned_f1 = -1.0
    best_quantized_f1 = -1.0
    global_epoch = 0

    if args.freeze_epochs > 0:
        stage = "frozen"
        configure_stage(model, stage)
        optimizer = make_optimizer(
            model,
            stage=stage,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            weight_decay=args.weight_decay,
        )
        scheduler = make_scheduler(
            optimizer,
            scheduler_name=args.lr_scheduler,
            inverse_decay=args.lr_aggression_decay,
            plateau_factor=args.lr_plateau_factor,
            plateau_patience=args.lr_plateau_patience,
            plateau_threshold=args.lr_plateau_threshold,
            min_lr=args.lr_min,
        )
        frozen_best_f1 = -1.0
        frozen_stale_epochs = 0

        for completed_frozen_epochs in range(1, args.freeze_epochs + 1):
            global_epoch += 1
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                stage=stage,
            )
            val_metrics = evaluate(model, val_loader, criterion, device)

            record = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": completed_frozen_epochs,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "train": train_metrics.to_dict(),
                "val": val_metrics.to_dict(),
            }
            history.append(record)
            save_json(args.output_dir / "metrics_history.json", history)

            print(
                f"epoch={global_epoch:03d} stage={stage} "
                f"train[{format_metrics(train_metrics)}] "
                f"val[{format_metrics(val_metrics)}]"
            )

            save_training_checkpoint(
                model,
                args=args,
                epoch=global_epoch,
                stage=stage,
                val_metrics=val_metrics,
                class_to_idx=train_dataset.class_to_idx,
                output_dir=args.output_dir,
                names=["last.pt"],
            )
            if val_metrics.f1 > best_f1:
                best_f1 = val_metrics.f1
                save_training_checkpoint(
                    model,
                    args=args,
                    epoch=global_epoch,
                    stage=stage,
                    val_metrics=val_metrics,
                    class_to_idx=train_dataset.class_to_idx,
                    output_dir=args.output_dir,
                    names=["best.pt"],
                )
            if improved(val_metrics.f1, frozen_best_f1, min_delta=args.min_delta):
                frozen_best_f1 = val_metrics.f1
                frozen_stale_epochs = 0
            else:
                frozen_stale_epochs += 1

            transition_reason = should_start_finetuning(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                stale_epochs=frozen_stale_epochs,
                completed_frozen_epochs=completed_frozen_epochs,
                max_frozen_epochs=args.freeze_epochs,
                min_finetune_f1=args.min_finetune_f1,
                max_head_train_f1=args.max_head_train_f1,
                frozen_patience=args.frozen_patience,
            )
            step_scheduler(scheduler, val_metrics)
            if transition_reason is not None:
                record["transition_to_finetune"] = transition_reason
                save_json(args.output_dir / "metrics_history.json", history)
                print(f"starting_finetune reason={transition_reason}")
                break

    fine_tune_stopped = False
    for stage, epochs in (("layer4", args.finetune_epochs), ("all", args.full_finetune_epochs)):
        if fine_tune_stopped or epochs <= 0:
            continue
        configure_stage(model, stage)
        optimizer = make_optimizer(
            model,
            stage=stage,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            weight_decay=args.weight_decay,
        )
        scheduler = make_scheduler(
            optimizer,
            scheduler_name=args.lr_scheduler,
            inverse_decay=args.lr_aggression_decay,
            plateau_factor=args.lr_plateau_factor,
            plateau_patience=args.lr_plateau_patience,
            plateau_threshold=args.lr_plateau_threshold,
            min_lr=args.lr_min,
        )
        stage_best_f1 = -1.0
        stage_stale_epochs = 0

        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                stage=stage,
            )
            val_metrics = evaluate(model, val_loader, criterion, device)

            record = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "train": train_metrics.to_dict(),
                "val": val_metrics.to_dict(),
            }
            quantized_val_metrics = None
            quantized_eval_record = None
            qat_reason = quantized_eval_reason(
                enabled=args.qat_eval,
                stage=stage,
                stage_epoch=stage_epoch,
                interval=args.qat_eval_interval,
                dense_after_stage_epoch=qat_dense_after_stage_epoch,
            )
            if qat_reason is not None:
                if qat_calibration_loader is None or qat_criterion is None:
                    raise RuntimeError("QAT eval was requested but calibration was not configured.")
                quantized_result = quantize_model_for_cpu(
                    model,
                    qat_calibration_loader,
                    image_size=args.image_size,
                    backend=args.qat_backend,
                    calibration_samples=args.qat_calibration_samples,
                )
                quantized_val_metrics = evaluate_model(
                    quantized_result.model,
                    val_loader,
                    qat_criterion,
                    cpu_device,
                    desc="val/int8",
                )
                quantized_eval_record = {
                    "reason": qat_reason,
                    "backend": quantized_result.backend,
                    "calibration_samples": quantized_result.calibration_samples,
                    "summary": quantized_result.summary,
                    "val": quantized_val_metrics.to_dict(),
                }
                record["quantized_eval"] = quantized_eval_record
                quantized_eval_history.append(
                    {
                        "epoch": global_epoch,
                        "stage": stage,
                        "stage_epoch": stage_epoch,
                        **quantized_eval_record,
                    }
                )
                save_json(args.output_dir / "quantized_eval_history.json", quantized_eval_history)
            elif args.qat_eval:
                record["quantized_eval_skipped"] = "scheduled_skip"

            history.append(record)
            save_json(args.output_dir / "metrics_history.json", history)

            message = (
                f"epoch={global_epoch:03d} stage={stage} "
                f"train[{format_metrics(train_metrics)}] "
                f"val[{format_metrics(val_metrics)}]"
            )
            if quantized_val_metrics is not None:
                message += f" int8_val[{format_metrics(quantized_val_metrics)}]"
            print(message)

            save_training_checkpoint(
                model,
                args=args,
                epoch=global_epoch,
                stage=stage,
                val_metrics=val_metrics,
                class_to_idx=train_dataset.class_to_idx,
                output_dir=args.output_dir,
                names=["last.pt"],
            )
            if val_metrics.f1 > best_f1:
                best_f1 = val_metrics.f1
                save_training_checkpoint(
                    model,
                    args=args,
                    epoch=global_epoch,
                    stage=stage,
                    val_metrics=val_metrics,
                    class_to_idx=train_dataset.class_to_idx,
                    output_dir=args.output_dir,
                    names=["best.pt"],
                )
            if val_metrics.f1 > best_finetuned_f1:
                best_finetuned_f1 = val_metrics.f1
                save_training_checkpoint(
                    model,
                    args=args,
                    epoch=global_epoch,
                    stage=stage,
                    val_metrics=val_metrics,
                    class_to_idx=train_dataset.class_to_idx,
                    output_dir=args.output_dir,
                    names=["best_finetuned.pt"],
                )
            if quantized_val_metrics is not None and quantized_val_metrics.f1 > best_quantized_f1:
                best_quantized_f1 = quantized_val_metrics.f1
                save_training_checkpoint(
                    model,
                    args=args,
                    epoch=global_epoch,
                    stage=stage,
                    val_metrics=val_metrics,
                    class_to_idx=train_dataset.class_to_idx,
                    output_dir=args.output_dir,
                    names=["best_quantized.pt"],
                    extra={
                        "quantized_val_metrics": quantized_val_metrics.to_dict(),
                        "quantized_eval": quantized_eval_record,
                    },
                )

            if improved(val_metrics.f1, stage_best_f1, min_delta=args.min_delta):
                stage_best_f1 = val_metrics.f1
                stage_stale_epochs = 0
            else:
                stage_stale_epochs += 1

            step_scheduler(scheduler, val_metrics)
            if args.finetune_patience > 0 and stage_stale_epochs >= args.finetune_patience:
                record["early_stop_reason"] = f"finetune_patience_{args.finetune_patience}"
                save_json(args.output_dir / "metrics_history.json", history)
                print(f"early_stop stage={stage} reason=finetune_patience_{args.finetune_patience}")
                fine_tune_stopped = True
                break

    if not history:
        raise ValueError("No training stages ran. Increase at least one epoch count.")
    message = f"best_val_f1={best_f1:.3f}"
    if args.qat_eval:
        message += f" best_quantized_val_f1={best_quantized_f1:.3f}"
    message += f" output_dir={args.output_dir}"
    print(message)


if __name__ == "__main__":
    main()
