"""Classify image folders with a trained crossing detector."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from crossing_detector.data import build_transforms
from crossing_detector.models import BACKBONE_SPECS, build_crossing_classifier


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class Prediction:
    source_path: Path
    predicted_class: str
    confidence: float
    probability_no_crossing: float
    probability_crossing: float


class ImagePredictionDataset(Dataset):
    def __init__(self, image_paths: list[Path], *, image_size: int) -> None:
        self.image_paths = image_paths
        self.transform = build_transforms(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/full-adaptive-gpu-qat-eval/best.pt"),
        help="Float32 training checkpoint. Ignored if --torchscript is set.",
    )
    model_group.add_argument("--torchscript", type=Path, default=None, help="TorchScript model, for example int8 export.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of images to classify.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Prediction output folder.")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--backbone", default=None, choices=sorted(BACKBONE_SPECS))
    parser.add_argument("--positive-class", default="crossing")
    parser.add_argument("--negative-class", default="no_crossing")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def find_image_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path must be a folder: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)


def load_float_checkpoint(
    checkpoint_path: Path,
    *,
    backbone: str | None,
    image_size: int | None,
    device: torch.device,
) -> tuple[nn.Module, int, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = payload.get("args", {})
    resolved_backbone = backbone or payload.get("backbone") or checkpoint_args.get("backbone") or "fmow-resnet50"
    resolved_image_size = image_size or int(checkpoint_args.get("image_size", 250))
    dropout = float(checkpoint_args.get("dropout", 0.2))
    model = build_crossing_classifier(
        resolved_backbone,
        pretrained=False,
        dropout=dropout,
        progress=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, resolved_image_size, {
        "checkpoint": str(checkpoint_path),
        "backbone": resolved_backbone,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_stage": payload.get("stage"),
    }


def load_torchscript_model(
    torchscript_path: Path,
    *,
    image_size: int | None,
    device: torch.device,
) -> tuple[nn.Module, int, dict]:
    model = torch.jit.load(str(torchscript_path), map_location=device)
    model.eval()
    return model, image_size or 250, {"torchscript": str(torchscript_path)}


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    positive_class: str,
    negative_class: str,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for images, paths in tqdm(loader, desc="predict", leave=False):
        logits = model(images.to(device))
        probabilities = torch.softmax(logits.detach().cpu(), dim=1)
        predicted_indices = torch.argmax(probabilities, dim=1)
        for path_text, probs, predicted_index in zip(paths, probabilities, predicted_indices):
            probability_no = float(probs[0].item())
            probability_yes = float(probs[1].item())
            predicted_class = positive_class if int(predicted_index.item()) == 1 else negative_class
            confidence = probability_yes if predicted_class == positive_class else probability_no
            predictions.append(
                Prediction(
                    source_path=Path(path_text),
                    predicted_class=predicted_class,
                    confidence=confidence,
                    probability_no_crossing=probability_no,
                    probability_crossing=probability_yes,
                )
            )
    return predictions


def write_predictions_csv(path: Path, predictions: Iterable[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "predicted_class",
                "confidence",
                "probability_no_crossing",
                "probability_crossing",
            ],
        )
        writer.writeheader()
        for item in predictions:
            writer.writerow(
                {
                    "source_path": str(item.source_path),
                    "predicted_class": item.predicted_class,
                    "confidence": f"{item.confidence:.8f}",
                    "probability_no_crossing": f"{item.probability_no_crossing:.8f}",
                    "probability_crossing": f"{item.probability_crossing:.8f}",
                }
            )


def copy_prediction_images(output_dir: Path, predictions: Iterable[Prediction]) -> None:
    for item in predictions:
        class_dir = output_dir / item.predicted_class
        class_dir.mkdir(parents=True, exist_ok=True)
        target = unique_output_path(class_dir, item.source_path.name)
        shutil.copy2(item.source_path, target)


def unique_output_path(folder: Path, filename: str) -> Path:
    candidate = folder / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        alternative = folder / f"{stem}_{index:03d}{suffix}"
        if not alternative.exists():
            return alternative
        index += 1


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.torchscript is not None:
        model, image_size, model_metadata = load_torchscript_model(
            args.torchscript,
            image_size=args.image_size,
            device=device,
        )
    else:
        model, image_size, model_metadata = load_float_checkpoint(
            args.checkpoint,
            backbone=args.backbone,
            image_size=args.image_size,
            device=device,
        )

    image_paths = find_image_paths(args.input_dir)
    if not image_paths:
        raise ValueError(f"No supported images found under {args.input_dir}.")

    dataset = ImagePredictionDataset(image_paths, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions = predict(
        model,
        loader,
        device=device,
        positive_class=args.positive_class,
        negative_class=args.negative_class,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(args.output_dir / "predictions.csv", predictions)
    if args.copy_images:
        copy_prediction_images(args.output_dir, predictions)

    positives = sum(item.predicted_class == args.positive_class for item in predictions)
    negatives = len(predictions) - positives
    print(
        "predicted "
        f"images={len(predictions)} "
        f"{args.positive_class}={positives} "
        f"{args.negative_class}={negatives} "
        f"device={device} "
        f"image_size={image_size} "
        f"model={model_metadata}"
    )


if __name__ == "__main__":
    main()
