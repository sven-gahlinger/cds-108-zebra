"""Prepare tracked evidence and optional large release assets for submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("runs/final-combined-gpu-qat-eval-continue")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_HANDIN_DIR = Path("handin/artifacts")
DEFAULT_RELEASE_DIR = Path("submission_assets")
RELEASE_TAG = "hackathon-submission-assets-v1"
DEFAULT_DATASET_PART_SIZE_GB = 1.8


TRACKED_ARTIFACTS = {
    "classmate_dataset_summary.json": Path("data/classmate_dataset_summary.json"),
    "own_dataset_summary.json": Path("data/own_dataset_summary.json"),
    "run_config.json": DEFAULT_RUN_ROOT / "run_config.json",
    "metrics_history.json": DEFAULT_RUN_ROOT / "metrics_history.json",
    "quantized_eval_history.json": DEFAULT_RUN_ROOT / "quantized_eval_history.json",
    "quantization_summary.json": DEFAULT_RUN_ROOT / "int8-best-quantized" / "quantization_summary.json",
    "test_metrics_float.json": DEFAULT_RUN_ROOT / "test_metrics_float.json",
}


LARGE_MODEL_ASSETS = {
    "best.pt": DEFAULT_RUN_ROOT / "best.pt",
    "best_quantized.pt": DEFAULT_RUN_ROOT / "best_quantized.pt",
    "model_int8_torchscript.pt": DEFAULT_RUN_ROOT / "int8-best-quantized" / "model_int8_torchscript.pt",
    "model_int8_state_dict.pt": DEFAULT_RUN_ROOT / "int8-best-quantized" / "model_int8_state_dict.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--handin-dir", type=Path, default=DEFAULT_HANDIN_DIR)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Copy model artifacts and create the dataset zip under submission_assets/.",
    )
    parser.add_argument(
        "--skip-dataset-archive",
        action="store_true",
        help="With --include-large, do not create the dataset ImageFolder zip.",
    )
    parser.add_argument(
        "--dataset-part-size-gb",
        type=float,
        default=DEFAULT_DATASET_PART_SIZE_GB,
        help="Split dataset archive into parts of this size for release uploads.",
    )
    parser.add_argument(
        "--keep-dataset-archive",
        action="store_true",
        help="Keep the unsplit dataset archive after creating .part files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handin_dir.mkdir(parents=True, exist_ok=True)
    args.release_dir.mkdir(parents=True, exist_ok=True)

    tracked = copy_tracked_artifacts(args.handin_dir, run_root=args.run_root, data_root=args.data_root)
    release_assets = inspect_large_assets(args.release_dir, run_root=args.run_root)

    if args.include_large:
        release_assets = copy_large_model_assets(args.release_dir, run_root=args.run_root)
        if not args.skip_dataset_archive:
            dataset_archive = args.release_dir / "hackathon_dataset_imagefolder.zip"
            cleanup_dataset_archives(dataset_archive)
            make_dataset_archive(args.data_root, dataset_archive)
            part_size = int(args.dataset_part_size_gb * 1024**3)
            dataset_assets = split_dataset_archive_if_needed(
                dataset_archive,
                part_size=part_size,
                keep_original=args.keep_dataset_archive,
            )
            release_assets.extend(dataset_assets)
        else:
            release_assets.extend(inspect_dataset_archive_assets(args.release_dir))

    manifest = {
        "release_tag": RELEASE_TAG,
        "strategy": (
            "Commit the small files in handin/artifacts/. Upload the large files in "
            "submission_assets/ as GitHub Release assets instead of committing them to Git."
        ),
        "tracked_artifacts": tracked,
        "release_assets": release_assets,
        "suggested_release_command": (
            "gh release create hackathon-submission-assets-v1 (Get-ChildItem submission_assets -File).FullName "
            "--title \"Hackathon submission assets\" "
            "--notes \"Model checkpoints, int8 export, and dataset archive for the cds-108 hackathon submission.\""
        ),
    }
    manifest_path = args.handin_dir / "submission_asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_artifacts_readme(args.handin_dir)
    print(f"Wrote tracked hand-in artifacts to {args.handin_dir}")
    print(f"Wrote asset manifest to {manifest_path}")
    if args.include_large:
        print(f"Wrote large release assets to {args.release_dir}")
    else:
        print("Large assets were only inspected. Rerun with --include-large to prepare release files.")


def copy_tracked_artifacts(handin_dir: Path, *, run_root: Path, data_root: Path) -> list[dict[str, Any]]:
    mapping = {
        name: resolve_artifact_source(source, run_root=run_root, data_root=data_root)
        for name, source in TRACKED_ARTIFACTS.items()
    }
    copied: list[dict[str, Any]] = []
    for name, source in mapping.items():
        target = handin_dir / name
        if not source.exists():
            copied.append({"name": name, "source": str(source), "exists": False})
            continue
        shutil.copy2(source, target)
        copied.append(describe_file(target, role="tracked_evidence", source=source))
    return copied


def inspect_large_assets(release_dir: Path, *, run_root: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for name, source in LARGE_MODEL_ASSETS.items():
        resolved = resolve_artifact_source(source, run_root=run_root, data_root=DEFAULT_DATA_ROOT)
        release_asset = release_dir / name
        if release_asset.exists():
            assets.append(describe_file(release_asset, role="large_model_asset", source=resolved))
        elif resolved.exists():
            assets.append(describe_file(resolved, role="large_model_asset"))
        else:
            assets.append({"name": name, "source": str(resolved), "exists": False})
    assets.extend(inspect_dataset_archive_assets(release_dir))
    return assets


def inspect_dataset_archive_assets(release_dir: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    dataset_archive = release_dir / "hackathon_dataset_imagefolder.zip"
    dataset_parts = sorted(release_dir.glob(f"{dataset_archive.name}.part*"))
    if dataset_parts:
        for part in dataset_parts:
            assets.append(describe_file(part, role="dataset_imagefolder_archive_part"))
    elif dataset_archive.exists():
        assets.append(describe_file(dataset_archive, role="dataset_imagefolder_archive"))
    else:
        assets.append(
            {
                "name": dataset_archive.name,
                "source": str(DEFAULT_DATA_ROOT),
                "exists": False,
                "role": "dataset_imagefolder_archive",
                "note": "Run with --include-large to create this archive.",
            }
        )
    return assets


def copy_large_model_assets(release_dir: Path, *, run_root: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for name, source in LARGE_MODEL_ASSETS.items():
        resolved = resolve_artifact_source(source, run_root=run_root, data_root=DEFAULT_DATA_ROOT)
        target = release_dir / name
        if not resolved.exists():
            copied.append({"name": name, "source": str(resolved), "exists": False})
            continue
        shutil.copy2(resolved, target)
        copied.append(describe_file(target, role="large_model_asset", source=resolved))
    return copied


def cleanup_dataset_archives(dataset_archive: Path) -> None:
    for path in [dataset_archive, *sorted(dataset_archive.parent.glob(f"{dataset_archive.name}.part*"))]:
        if path.exists():
            path.unlink()


def split_dataset_archive_if_needed(
    dataset_archive: Path,
    *,
    part_size: int,
    keep_original: bool,
) -> list[dict[str, Any]]:
    if part_size <= 0 or dataset_archive.stat().st_size <= part_size:
        return [describe_file(dataset_archive, role="dataset_imagefolder_archive")]

    parts = split_large_file(dataset_archive, part_size=part_size)
    described = [
        describe_file(part, role="dataset_imagefolder_archive_part", source=dataset_archive)
        for part in parts
    ]
    described.append(
        {
            "name": dataset_archive.name,
            "source": str(dataset_archive),
            "role": "dataset_imagefolder_archive",
            "exists": False,
            "note": (
                "Archive was split into .part files for upload. Reassemble the parts in "
                "filename order to recover this zip."
            ),
            "parts": [part.name for part in parts],
        }
    )
    if not keep_original:
        dataset_archive.unlink()
    return described


def split_large_file(path: Path, *, part_size: int) -> list[Path]:
    parts: list[Path] = []
    block_size = 1024 * 1024 * 8
    with path.open("rb") as source:
        index = 1
        while True:
            part = path.with_name(f"{path.name}.part{index:03d}")
            written = 0
            with part.open("wb") as target:
                while written < part_size:
                    chunk = source.read(min(block_size, part_size - written))
                    if not chunk:
                        break
                    target.write(chunk)
                    written += len(chunk)
            if written == 0:
                part.unlink(missing_ok=True)
                break
            parts.append(part)
            index += 1
    return parts


def resolve_artifact_source(path: Path, *, run_root: Path, data_root: Path) -> Path:
    text = str(path)
    text = text.replace(str(DEFAULT_RUN_ROOT), str(run_root))
    text = text.replace(str(DEFAULT_DATA_ROOT), str(data_root), 1)
    return Path(text)


def make_dataset_archive(data_root: Path, target: Path) -> None:
    required = [data_root / split for split in ("train", "val", "test")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot archive dataset; missing folders: {', '.join(missing)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for split in ("train", "val", "test"):
            add_tree_to_zip(archive, data_root / split, root=data_root)
        for metadata_name in (
            "classmate_dataset_summary.json",
            "classmate_dataset_manifest.csv",
            "own_dataset_summary.json",
            "own_dataset_manifest.csv",
        ):
            metadata_path = data_root / metadata_name
            if metadata_path.exists():
                archive.write(metadata_path, metadata_path.relative_to(data_root.parent))


def add_tree_to_zip(archive: zipfile.ZipFile, folder: Path, *, root: Path) -> None:
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(root.parent))


def describe_file(path: Path, *, role: str, source: Path | None = None) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": str(path),
        "source": None if source is None else str(source),
        "role": role,
        "exists": path.exists(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifacts_readme(handin_dir: Path) -> None:
    path = handin_dir / "README.md"
    path.write_text(
        """# Hand-In Artifacts

This folder contains small, tracked evidence files for the hackathon report.

Large binary assets are intentionally not committed to Git:

- float32 model checkpoints (`best.pt`, `best_quantized.pt`)
- int8 TorchScript/state-dict exports
- dataset image archive, split into `.partNNN` files when it is large

Use `uv run python scripts/prepare_submission_assets.py --include-large` to
prepare those large files under `submission_assets/`, then upload them as GitHub
Release assets. The generated `submission_asset_manifest.json` contains file
sizes and SHA-256 checksums so the release assets can be verified. If the
dataset archive was split, reassemble the `.partNNN` files in filename order to
recover `hackathon_dataset_imagefolder.zip`.

Windows example after downloading all parts into one folder:

```powershell
cmd /c copy /b hackathon_dataset_imagefolder.zip.part001+hackathon_dataset_imagefolder.zip.part002+hackathon_dataset_imagefolder.zip.part003 hackathon_dataset_imagefolder.zip
```
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
