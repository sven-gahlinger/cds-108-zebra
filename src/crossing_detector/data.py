"""Dataset and transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataRoots:
    train: Path
    val: Path


class BinaryImageFolder(Dataset):
    """ImageFolder wrapper that maps a named class to positive label 1."""

    def __init__(self, root: str | Path, *, transform, positive_class: str) -> None:
        from torchvision.datasets import ImageFolder

        self.root = Path(root)
        self.dataset = ImageFolder(self.root, transform=transform)
        self.positive_class = positive_class
        if len(self.dataset.classes) != 2:
            raise ValueError(
                f"{self.root} must contain exactly two class folders; "
                f"found {self.dataset.classes}."
            )
        if positive_class not in self.dataset.class_to_idx:
            raise ValueError(
                f"Positive class '{positive_class}' not found in {self.root}. "
                f"Available classes: {self.dataset.classes}."
            )
        self.positive_index = self.dataset.class_to_idx[positive_class]

    @property
    def classes(self) -> list[str]:
        return list(self.dataset.classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return dict(self.dataset.class_to_idx)

    @property
    def label_counts(self) -> dict[str, int]:
        counts = {"negative": 0, "positive": 0}
        for _, class_index in self.dataset.samples:
            if class_index == self.positive_index:
                counts["positive"] += 1
            else:
                counts["negative"] += 1
        return counts

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, class_index = self.dataset[index]
        label = 1 if class_index == self.positive_index else 0
        return image, torch.tensor(label, dtype=torch.long)


def build_transforms(*, image_size: int, train: bool):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    common = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
    ]
    if train:
        common.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180, interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            ]
        )
    common.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD),
        ]
    )
    return transforms.Compose(common)


def resolve_data_roots(
    *,
    data_root: str | Path | None,
    train_root: str | Path | None,
    val_root: str | Path | None,
) -> DataRoots:
    if data_root is not None:
        base = Path(data_root)
        resolved = DataRoots(train=base / "train", val=base / "val")
    elif train_root is not None and val_root is not None:
        resolved = DataRoots(train=Path(train_root), val=Path(val_root))
    else:
        raise ValueError("Provide either --data-root or both --train-root and --val-root.")

    missing = [str(path) for path in (resolved.train, resolved.val) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset folder(s): {', '.join(missing)}")
    return resolved


def build_dataloaders(
    roots: DataRoots,
    *,
    positive_class: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, BinaryImageFolder, BinaryImageFolder]:
    train_dataset = BinaryImageFolder(
        roots.train,
        transform=build_transforms(image_size=image_size, train=True),
        positive_class=positive_class,
    )
    val_dataset = BinaryImageFolder(
        roots.val,
        transform=build_transforms(image_size=image_size, train=False),
        positive_class=positive_class,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_dataset, val_dataset
