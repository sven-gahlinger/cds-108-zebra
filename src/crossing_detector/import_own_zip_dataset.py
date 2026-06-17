"""Import Sven's labeled zebra-crossing zip into the project ImageFolder layout."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


CLASS_MAP = {
    "y": "crossing",
    "n": "no_crossing",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ZipImage:
    zip_path: str
    source_class: str
    target_class: str
    filename: str
    length: int


@dataclass(frozen=True)
class ImportedImage:
    source_zip_path: str
    split: str
    target_class: str
    output_path: str
    bytes: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import zebra/y and zebra/n images into data/{split}/{class}."
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/svenzebradata.zip"),
        help="Zip file containing zebra/y and zebra/n folders.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root containing train/val/test class folders.",
    )
    parser.add_argument(
        "--prefix",
        default="own_svenzebradata",
        help="Filename prefix for imported files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/own_dataset_manifest.csv"),
        help="CSV manifest to write.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/own_dataset_summary.json"),
        help="JSON summary to write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing imported images.",
    )
    return parser.parse_args()


def iter_zip_images(zip_file: ZipFile) -> list[ZipImage]:
    images: list[ZipImage] = []
    for entry in zip_file.infolist():
        if entry.is_dir():
            continue
        path = PurePosixPath(entry.filename)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.parts
        source_class = None
        if len(parts) >= 3 and parts[0] == "zebra":
            source_class = parts[1]
        if source_class not in CLASS_MAP:
            continue
        images.append(
            ZipImage(
                zip_path=entry.filename,
                source_class=source_class,
                target_class=CLASS_MAP[source_class],
                filename=path.name,
                length=entry.file_size,
            )
        )
    return images


def split_images(
    images: list[ZipImage],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[ZipImage]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive.")

    train_ratio = train_ratio / ratio_sum
    val_ratio = val_ratio / ratio_sum
    rng = random.Random(seed)

    by_class: dict[str, list[ZipImage]] = {}
    for image in images:
        by_class.setdefault(image.target_class, []).append(image)

    split_map: dict[str, list[ZipImage]] = {"train": [], "val": [], "test": []}
    for class_images in by_class.values():
        shuffled = sorted(class_images, key=lambda image: image.zip_path)
        rng.shuffle(shuffled)
        total = len(shuffled)

        train_count = round(total * train_ratio)
        val_count = round(total * val_ratio)
        if total >= 3 and val_ratio > 0 and val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if total >= 3 and test_ratio > 0 and test_count == 0:
            test_count = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1
        if test_count < 0:
            train_count = max(0, train_count + test_count)
            test_count = 0

        split_map["train"].extend(shuffled[:train_count])
        split_map["val"].extend(shuffled[train_count : train_count + val_count])
        split_map["test"].extend(shuffled[train_count + val_count :])

    for split_images_for_class in split_map.values():
        split_images_for_class.sort(key=lambda image: image.zip_path)
    return split_map


def make_output_path(data_root: Path, split: str, target_class: str, prefix: str, filename: str) -> Path:
    return data_root / split / target_class / f"{prefix}_{filename}"


def extract_images(
    zip_file: ZipFile,
    split_map: dict[str, list[ZipImage]],
    data_root: Path,
    prefix: str,
    overwrite: bool,
) -> list[ImportedImage]:
    imported: list[ImportedImage] = []
    for split, images in split_map.items():
        for image in images:
            output_path = make_output_path(
                data_root=data_root,
                split=split,
                target_class=image.target_class,
                prefix=prefix,
                filename=image.filename,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            status = "written"
            if output_path.exists() and not overwrite:
                status = "skipped_existing"
            else:
                with zip_file.open(image.zip_path) as source, output_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            imported.append(
                ImportedImage(
                    source_zip_path=image.zip_path,
                    split=split,
                    target_class=image.target_class,
                    output_path=str(output_path),
                    bytes=image.length,
                    status=status,
                )
            )
    return imported


def write_manifest(path: Path, rows: list[ImportedImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(
    path: Path,
    *,
    zip_path: Path,
    data_root: Path,
    prefix: str,
    rows: list[ImportedImage],
    seed: int,
    ratios: dict[str, float],
) -> None:
    split_class_counts = Counter((row.split, row.target_class) for row in rows)
    class_counts = Counter(row.target_class for row in rows)
    status_counts = Counter(row.status for row in rows)
    summary = {
        "source_zip": str(zip_path),
        "data_root": str(data_root),
        "filename_prefix": prefix,
        "seed": seed,
        "split_ratios": ratios,
        "total_images": len(rows),
        "total_bytes": sum(row.bytes for row in rows),
        "class_counts": dict(sorted(class_counts.items())),
        "split_class_counts": {
            f"{split}/{target_class}": split_class_counts[(split, target_class)]
            for split in ("train", "val", "test")
            for target_class in ("crossing", "no_crossing")
        },
        "status_counts": dict(sorted(status_counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(args.zip)

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    with ZipFile(args.zip) as zip_file:
        images = iter_zip_images(zip_file)
        if not images:
            raise RuntimeError(f"No labeled images found in {args.zip}.")
        split_map = split_images(
            images=images,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        rows = extract_images(
            zip_file=zip_file,
            split_map=split_map,
            data_root=args.data_root,
            prefix=args.prefix,
            overwrite=args.overwrite,
        )

    write_manifest(args.manifest, rows)
    write_summary(
        args.summary,
        zip_path=args.zip,
        data_root=args.data_root,
        prefix=args.prefix,
        rows=rows,
        seed=args.seed,
        ratios=ratios,
    )
    counts = Counter((row.split, row.target_class) for row in rows)
    print(f"Imported {len(rows)} images from {args.zip}")
    for split in ("train", "val", "test"):
        print(
            f"{split}: crossing={counts[(split, 'crossing')]} "
            f"no_crossing={counts[(split, 'no_crossing')]}"
        )
    print(f"Manifest: {args.manifest}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
