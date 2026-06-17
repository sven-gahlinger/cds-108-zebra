"""Post-training static int8 quantization for the crossing classifier."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader
from tqdm import tqdm

from crossing_detector.data import BinaryImageFolder, build_transforms, resolve_data_roots
from crossing_detector.metrics import BinaryMetrics, compute_binary_metrics
from crossing_detector.models import BACKBONE_SPECS, build_crossing_classifier


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class QuantizedModelResult:
    model: nn.Module
    backend: str
    calibration_samples: int
    summary: dict[str, Any]


class TraceableCrossingClassifier(nn.Module):
    """FX-traceable inference wrapper for timm ResNet backbones.

    The training model keeps defensive shape checks in ``forward``. FX static
    quantization cannot trace those branches, so this wrapper fixes the known
    ResNet path: ``forward_features -> global average pool -> MLP head``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.pool = model.pool
        self.head = model.head

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone.forward_features(images)
        features = self.pool(features).flatten(1)
        return self.head(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/full-adaptive-gpu/best_finetuned.pt"),
        help="Float32 training checkpoint to quantize.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Folder containing train/ and val/.")
    parser.add_argument("--train-root", type=Path, default=None, help="Explicit training ImageFolder root.")
    parser.add_argument("--val-root", type=Path, default=None, help="Explicit validation ImageFolder root.")
    parser.add_argument("--test-root", type=Path, default=None, help="Optional test ImageFolder root.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--positive-class", default=None)
    parser.add_argument("--backbone", default=None, choices=sorted(BACKBONE_SPECS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--calibration-split", choices=("train", "val"), default="train")
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--backend", default="auto", help="Quantized CPU backend, or 'auto'.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_quantized_backend(requested: str = "auto") -> str:
    supported = list(torch.backends.quantized.supported_engines)
    if requested != "auto":
        if requested not in supported:
            raise ValueError(
                f"Quantized backend '{requested}' is not supported by this PyTorch build. "
                f"Supported: {supported}"
            )
        return requested

    for candidate in ("x86", "fbgemm", "onednn", "qnnpack"):
        if candidate in supported:
            return candidate
    if supported:
        return supported[0]
    raise RuntimeError("This PyTorch build does not report any quantized CPU backend.")


def load_float_model(
    checkpoint_path: Path,
    *,
    backbone: str | None,
    dropout: float | None,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = payload.get("args", {})
    resolved_backbone = backbone or payload.get("backbone") or checkpoint_args.get("backbone") or "fmow-resnet50"
    resolved_dropout = dropout if dropout is not None else float(checkpoint_args.get("dropout", 0.2))

    model = build_crossing_classifier(
        resolved_backbone,
        pretrained=False,
        dropout=resolved_dropout,
        progress=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_stage": payload.get("stage"),
        "checkpoint_val_metrics": payload.get("val_metrics"),
        "positive_class": payload.get("positive_class"),
        "backbone": resolved_backbone,
        "dropout": resolved_dropout,
    }
    return model, metadata


@torch.inference_mode()
def calibrate(model: nn.Module, loader: DataLoader, *, max_samples: int, device: torch.device) -> int:
    model.eval()
    seen = 0
    progress = tqdm(loader, desc="calibrate/int8", leave=False)
    for images, _ in progress:
        if max_samples > 0 and seen >= max_samples:
            break
        remaining = max_samples - seen if max_samples > 0 else images.shape[0]
        if max_samples > 0 and images.shape[0] > remaining:
            images = images[:remaining]
        model(images.to(device))
        seen += images.shape[0]
        progress.set_postfix(samples=seen)
    return seen


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    desc: str = "val",
) -> BinaryMetrics:
    model.eval()
    total_loss = 0.0
    seen = 0
    logits_all: list[Tensor] = []
    labels_all: list[Tensor] = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
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


def make_eval_loader(
    root: Path,
    *,
    image_size: int,
    positive_class: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, BinaryImageFolder]:
    dataset = BinaryImageFolder(
        root,
        transform=build_transforms(image_size=image_size, train=False),
        positive_class=positive_class,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader, dataset


def make_calibration_loader(
    root: Path,
    *,
    image_size: int,
    positive_class: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = BinaryImageFolder(
        root,
        transform=build_transforms(image_size=image_size, train=False),
        positive_class=positive_class,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )


def module_quantization_summary(model: nn.Module) -> dict[str, Any]:
    quantized_by_kind: Counter[str] = Counter()
    weight_dtypes: Counter[str] = Counter()

    for module in model.modules():
        module_path = type(module).__module__.lower()
        module_name = type(module).__name__
        if "quantized" in module_path:
            if "conv" in module_name.lower():
                quantized_by_kind["conv"] += 1
            elif "linear" in module_name.lower():
                quantized_by_kind["linear"] += 1
            else:
                quantized_by_kind["other"] += 1

        weight = getattr(module, "weight", None)
        if callable(weight):
            try:
                weight = weight()
            except RuntimeError:
                weight = None
        if torch.is_tensor(weight):
            weight_dtypes[str(weight.dtype)] += 1

    return {
        "quantized_modules": dict(quantized_by_kind),
        "module_weight_dtypes": dict(weight_dtypes),
    }


def quantize_model_for_cpu(
    model: nn.Module,
    calibration_loader: DataLoader,
    *,
    image_size: int,
    backend: str = "auto",
    calibration_samples: int = 256,
) -> QuantizedModelResult:
    """Create a CPU int8 shadow model from a float32 model."""

    resolved_backend = choose_quantized_backend(backend)
    torch.backends.quantized.engine = resolved_backend
    device = torch.device("cpu")
    float_copy = copy.deepcopy(model).to(device).eval()
    quantizable_model = TraceableCrossingClassifier(float_copy).eval()
    example_input = (torch.randn(1, 3, image_size, image_size),)
    qconfig_mapping = get_default_qconfig_mapping(resolved_backend)
    prepared_model = prepare_fx(quantizable_model, qconfig_mapping, example_input)
    calibration_count = calibrate(
        prepared_model,
        calibration_loader,
        max_samples=calibration_samples,
        device=device,
    )
    quantized_model = convert_fx(prepared_model).eval()
    return QuantizedModelResult(
        model=quantized_model,
        backend=resolved_backend,
        calibration_samples=calibration_count,
        summary=module_quantization_summary(quantized_model),
    )


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    backend = choose_quantized_backend(args.backend)

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir or checkpoint_path.parent / "int8"
    output_dir.mkdir(parents=True, exist_ok=True)

    float_model, checkpoint_metadata = load_float_model(
        checkpoint_path,
        backbone=args.backbone,
        dropout=args.dropout,
    )
    checkpoint_args = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("args", {})
    image_size = args.image_size or int(checkpoint_args.get("image_size", 250))
    positive_class = (
        args.positive_class
        or checkpoint_metadata.get("positive_class")
        or checkpoint_args.get("positive_class")
        or "crossing"
    )

    data_root = args.data_root
    if data_root is None and (args.train_root is None or args.val_root is None):
        data_root = Path("data")
    roots = resolve_data_roots(
        data_root=data_root,
        train_root=args.train_root,
        val_root=args.val_root,
    )
    calibration_root = roots.train if args.calibration_split == "train" else roots.val
    calibration_loader = make_calibration_loader(
        calibration_root,
        image_size=image_size,
        positive_class=positive_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    example_input = (torch.randn(1, 3, image_size, image_size),)
    quantized_result = quantize_model_for_cpu(
        float_model,
        calibration_loader,
        image_size=image_size,
        backend=backend,
        calibration_samples=args.calibration_samples,
    )
    quantized_model = quantized_result.model

    criterion = nn.CrossEntropyLoss()
    val_loader, val_dataset = make_eval_loader(
        roots.val,
        image_size=image_size,
        positive_class=positive_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_metrics = evaluate_model(quantized_model, val_loader, criterion, device)

    test_metrics = None
    test_root = args.test_root or (data_root / "test" if data_root is not None else None)
    if test_root is not None and test_root.exists():
        test_loader, _ = make_eval_loader(
            test_root,
            image_size=image_size,
            positive_class=positive_class,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_metrics = evaluate_model(quantized_model, test_loader, criterion, device)

    artifact_metadata = {
        **checkpoint_metadata,
        "image_size": image_size,
        "positive_class": positive_class,
        "backend": backend,
    }

    torchscript_path = output_dir / "model_int8_torchscript.pt"
    torchscript_error = None
    try:
        traced = torch.jit.trace(quantized_model, example_input)
        traced.save(str(torchscript_path))
    except Exception as exc:  # pragma: no cover - depends on PyTorch backend support
        torchscript_path = None
        torchscript_error = repr(exc)

    graphmodule_path = output_dir / "model_int8_fx.pt"
    graphmodule_error = None
    try:
        torch.save(
            {
                "model": quantized_model,
                "metadata": artifact_metadata,
            },
            graphmodule_path,
        )
    except Exception as exc:  # pragma: no cover - depends on PyTorch backend support
        if graphmodule_path.exists():
            graphmodule_path.unlink()
        graphmodule_path = None
        graphmodule_error = repr(exc)

    state_dict_path = output_dir / "model_int8_state_dict.pt"
    torch.save(
        {
            "state_dict": quantized_model.state_dict(),
            "metadata": artifact_metadata,
            "note": (
                "Quantized state dict only. Recreate the FX-quantized graph with the "
                "same model/backbone/backend before loading this state dict."
            ),
        },
        state_dict_path,
    )

    summary = {
        "quantization": {
            "method": "post_training_static_fx",
            "dtype": "int8 weights / quantized activations, float32 output logits",
            "device": "cpu",
            "backend": quantized_result.backend,
            "calibration_split": args.calibration_split,
            "calibration_samples_requested": args.calibration_samples,
            "calibration_samples_used": quantized_result.calibration_samples,
            "summary": quantized_result.summary,
        },
        "checkpoint": checkpoint_metadata,
        "data": {
            "train_root": str(roots.train),
            "val_root": str(roots.val),
            "test_root": None if test_root is None else str(test_root),
            "val_counts": val_dataset.label_counts,
            "image_size": image_size,
            "positive_class": positive_class,
        },
        "metrics": {
            "val": val_metrics.to_dict(),
            "test": None if test_metrics is None else test_metrics.to_dict(),
        },
        "artifacts": {
            "fx_graphmodule": None if graphmodule_path is None else str(graphmodule_path),
            "fx_graphmodule_error": graphmodule_error,
            "state_dict": str(state_dict_path),
            "torchscript": None if torchscript_path is None else str(torchscript_path),
            "torchscript_error": torchscript_error,
        },
    }
    save_json(output_dir / "quantization_summary.json", summary)

    print(
        "int8_quantized "
        f"backend={quantized_result.backend} "
        f"calibration_samples={quantized_result.calibration_samples} "
        f"val_f1={val_metrics.f1:.3f} "
        f"val_acc={val_metrics.accuracy:.3f} "
        f"output_dir={output_dir}"
    )
    if test_metrics is not None:
        print(
            "int8_test "
            f"f1={test_metrics.f1:.3f} "
            f"acc={test_metrics.accuracy:.3f} "
            f"precision={test_metrics.precision:.3f} "
            f"recall={test_metrics.recall:.3f}"
        )


if __name__ == "__main__":
    main()
