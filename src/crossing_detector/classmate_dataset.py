"""Prepare our ImageFolder dataset from DotNaos' released metadata-only dataset.

This uses only the dataset acquisition/export path from:
https://github.com/DotNaos/fs26-crosswalk-detector
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import ssl
import tarfile
import tempfile
from time import monotonic
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
from PIL import Image


RELEASE_DATASET_URL = (
    "https://github.com/DotNaos/fs26-crosswalk-detector/releases/download/"
    "submission-dataset-v1/sam3-500k-static-metadata-v1.tar.gz"
)
RELEASE_DATASET_SHA256 = "1a51b3e25661acd322f6a183a82a33a5201b68da72a20c64f99e8cef4ba10993"
DATASET_ID = "sam3-500k-masks-v1"
EARTH_RADIUS_M = 6_378_137.0
WMS_BASE_URL = "https://wms.geo.admin.ch/"
USER_AGENT = "swissimage-crossing-detector/0.1 dataset-import"


@dataclass(frozen=True)
class Candidate:
    image_id: str
    tile_id: str
    scene_id: str
    city: str
    row: int
    col: int
    bbox_mercator: tuple[float, float, float, float]
    relative_path: str
    label: str
    confidence: float
    mask_path: str
    mask_coverage: float


@dataclass(frozen=True)
class SceneRequest:
    scene_id: str
    latitude: float
    longitude: float
    size_m: int
    image_px: int
    tile_size_m: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ImageFolder train/val/test data from the classmate dataset metadata."
    )
    parser.add_argument("--metadata-root", type=Path, default=Path("data/source/classmate-static-datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/classmate"))
    parser.add_argument("--image-size", type=int, default=250)
    parser.add_argument("--positive-limit", type=int, default=2500)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--min-mask-coverage", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-metadata-download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = ensure_metadata_dataset(
        args.metadata_root,
        skip_download=args.skip_metadata_download,
        show_progress=not args.no_progress,
    )
    summary = prepare_imagefolder_export(
        dataset_root,
        args.output_dir,
        raw_root=args.raw_root,
        image_size=args.image_size,
        positive_limit=args.positive_limit,
        negative_ratio=args.negative_ratio,
        min_confidence=args.min_confidence,
        min_mask_coverage=args.min_mask_coverage,
        seed=args.seed,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2))


def ensure_metadata_dataset(
    metadata_root: Path,
    *,
    skip_download: bool,
    show_progress: bool,
) -> Path:
    dataset_root = metadata_root / DATASET_ID
    if (dataset_root / "dataset.json").exists():
        return dataset_root
    if skip_download:
        raise FileNotFoundError(f"Missing metadata dataset: {dataset_root}")

    metadata_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="classmate-dataset-") as temporary_dir:
        archive_path = Path(temporary_dir) / "dataset.tar.gz"
        download_file(RELEASE_DATASET_URL, archive_path, RELEASE_DATASET_SHA256, show_progress=show_progress)
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract(archive, metadata_root, show_progress=show_progress)
    if not (dataset_root / "dataset.json").exists():
        raise FileNotFoundError(f"Release archive did not create {dataset_root}")
    return dataset_root


def prepare_imagefolder_export(
    dataset_root: Path,
    output_dir: Path,
    *,
    raw_root: Path,
    image_size: int,
    positive_limit: int,
    negative_ratio: float,
    min_confidence: float,
    min_mask_coverage: float,
    seed: int,
    overwrite: bool,
    show_progress: bool,
) -> dict[str, Any]:
    positives, negatives = load_candidates(
        dataset_root,
        min_confidence=min_confidence,
        min_mask_coverage=min_mask_coverage,
    )
    positives = sorted(positives, key=lambda item: item.confidence, reverse=True)[:positive_limit]
    selected_negatives = select_negatives(positives, negatives, negative_ratio=negative_ratio, seed=seed)
    selected = positives + selected_negatives
    random.Random(seed).shuffle(selected)
    split_by_tile_id = stratified_splits(selected)

    if overwrite:
        for name in ("train", "val", "test"):
            target = output_dir / name
            if target.exists():
                shutil.rmtree(target)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scene_cache: dict[str, Image.Image] = {}
    total = len(selected)
    started_at = monotonic()
    for index, candidate in enumerate(selected, start=1):
        split = split_by_tile_id[candidate.tile_id]
        class_name = "crossing" if candidate.label == "crosswalk" else "no_crossing"
        target_dir = output_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"{safe_name(candidate.image_id)}.jpg"
        image_path = target_dir / image_name
        if not image_path.exists():
            scene = scene_request(dataset_root, candidate.scene_id)
            scene_image = scene_cache.setdefault(
                candidate.scene_id,
                load_cached_scene_image(dataset_root, scene, raw_root=raw_root),
            )
            crop_tile(scene_image, scene, candidate.bbox_mercator).resize(
                (image_size, image_size), Image.Resampling.BICUBIC
            ).save(image_path, quality=94)
        rows.append(
            {
                "image_path": str(image_path),
                "split": split,
                "class_name": class_name,
                "source_label": candidate.label,
                "tile_id": candidate.tile_id,
                "scene_id": candidate.scene_id,
                "city": candidate.city,
                "confidence": round(candidate.confidence, 6),
                "mask_coverage": round(candidate.mask_coverage, 6),
            }
        )
        if show_progress and should_emit_progress(index, total):
            elapsed = monotonic() - started_at
            print(f"Prepared {index}/{total} images in {elapsed:.1f}s", flush=True)

    manifest_path = output_dir / "classmate_dataset_manifest.csv"
    write_manifest(manifest_path, rows)
    summary = {
        "source_repo": "https://github.com/DotNaos/fs26-crosswalk-detector",
        "source_release": RELEASE_DATASET_URL,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "raw_root": str(raw_root),
        "image_size": image_size,
        "positive_limit": positive_limit,
        "negative_ratio": negative_ratio,
        "min_confidence": min_confidence,
        "min_mask_coverage": min_mask_coverage,
        "seed": seed,
        "samples": len(rows),
        "counts": count_rows(rows),
        "manifest": str(manifest_path),
    }
    (output_dir / "classmate_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def download_file(url: str, target: Path, expected_sha256: str, *, show_progress: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    if show_progress:
        print(f"Downloading {url}", flush=True)
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_report = monotonic()
        with temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = monotonic()
                if show_progress and now - last_report >= 1.0:
                    print(f"  {format_bytes(downloaded)} / {format_bytes(total)}", flush=True)
                    last_report = now
    verify_sha256(temporary, expected_sha256)
    temporary.replace(target)


def safe_extract(archive: tarfile.TarFile, destination: Path, *, show_progress: bool) -> None:
    root = destination.resolve()
    members = archive.getmembers()
    for index, member in enumerate(members, start=1):
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"Archive member escapes destination: {member.name}")
        archive.extract(member, destination, filter="data")
        if show_progress and (index == 1 or index == len(members) or index % 500 == 0):
            print(f"  extracted {index}/{len(members)}", flush=True)


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def load_candidates(
    dataset_root: Path,
    *,
    min_confidence: float,
    min_mask_coverage: float,
) -> tuple[list[Candidate], list[Candidate]]:
    index = read_json(dataset_root / "dataset.json")
    positives: list[Candidate] = []
    negatives: list[Candidate] = []
    for shard in index["shards"]:
        for row in read_jsonl(resolve_shard_path(dataset_root, shard["path"])):
            resolved = row.get("resolved_label", {})
            label = str(resolved.get("decision", ""))
            confidence = float(resolved.get("confidence") or 0.0)
            candidate = candidate_from_row(dataset_root, row, label, confidence)
            if candidate is None:
                continue
            if label == "crosswalk":
                if confidence >= min_confidence and candidate.mask_coverage >= min_mask_coverage:
                    positives.append(candidate)
            elif label == "no_crosswalk":
                negatives.append(candidate)
    return positives, negatives


def candidate_from_row(dataset_root: Path, row: dict[str, Any], label: str, confidence: float) -> Candidate | None:
    mask_path = ""
    coverage = 0.0
    if label == "crosswalk":
        mask_path = row_mask_path(dataset_root, row)
        if not mask_path or not Path(mask_path).exists():
            return None
        coverage = mask_coverage(Path(mask_path))
    if label not in {"crosswalk", "no_crosswalk"}:
        return None
    return Candidate(
        image_id=str(row["image_id"]),
        tile_id=str(row["tile_id"]),
        scene_id=str(row["scene_id"]),
        city=str(row.get("city", "")),
        row=int(row["row"]),
        col=int(row["col"]),
        bbox_mercator=tuple(float(value) for value in row["bbox_mercator"]),
        relative_path=str(row["reconstruction"]["relative_path"]),
        label=label,
        confidence=confidence,
        mask_path=mask_path,
        mask_coverage=coverage,
    )


def select_negatives(
    positives: list[Candidate],
    negatives: list[Candidate],
    *,
    negative_ratio: float,
    seed: int,
) -> list[Candidate]:
    target_count = round(len(positives) * negative_ratio)
    positive_scenes = {candidate.scene_id for candidate in positives}
    same_scene_negatives = [candidate for candidate in negatives if candidate.scene_id in positive_scenes]
    pool = same_scene_negatives if len(same_scene_negatives) >= target_count else negatives
    count = min(len(pool), target_count)
    return random.Random(seed).sample(pool, count)


def row_mask_path(dataset_root: Path, row: dict[str, Any]) -> str:
    resolved_source = row.get("resolved_label", {}).get("source_id")
    candidates: list[tuple[str | None, str]] = []
    for label in row.get("labels", []):
        metadata = label.get("metadata") if isinstance(label, dict) else None
        if not isinstance(metadata, dict):
            continue
        artifact = metadata.get("mask_artifact")
        if isinstance(artifact, dict) and artifact.get("path"):
            path = resolve_mask_artifact_path(dataset_root, artifact)
            if path:
                source_id = label.get("source", {}).get("source_id")
                candidates.append((source_id, path))
    for source_id, path in candidates:
        if source_id == resolved_source:
            return path
    return candidates[-1][1] if candidates else ""


def resolve_mask_artifact_path(dataset_root: Path, artifact: dict[str, Any]) -> str:
    static_path = artifact.get("static_path")
    if static_path:
        candidate = dataset_root / str(static_path)
        if candidate.exists():
            return str(candidate)
    path = artifact.get("path")
    if path and Path(str(path)).exists():
        return str(path)
    return ""


def mask_coverage(path: Path) -> float:
    mask = Image.open(path).convert("L")
    histogram = mask.histogram()
    total = sum(histogram)
    if total == 0:
        return 0.0
    return (total - histogram[0]) / total


def load_cached_scene_image(dataset_root: Path, scene: SceneRequest, *, raw_root: Path) -> Image.Image:
    scene_root = raw_root / DATASET_ID / "wms-mosaics"
    scene_root.mkdir(parents=True, exist_ok=True)
    path = scene_root / f"{scene.scene_id}.jpg"
    if not path.exists() or path.stat().st_size == 0:
        payload = fetch_url(wms_url(scene))
        temporary = path.with_suffix(".jpg.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return Image.open(path).convert("RGB")


def scene_request(dataset_root: Path, scene_id: str) -> SceneRequest:
    scene_path = dataset_root / "scenes" / scene_id / "scene.json"
    if scene_path.exists():
        scene = read_json(scene_path)
        return SceneRequest(
            scene_id=scene_id,
            latitude=float(scene["latitude"]),
            longitude=float(scene["longitude"]),
            size_m=int(scene["size_m"]),
            image_px=int(scene["image_px"]),
            tile_size_m=int(scene["tile_size_m"]),
        )

    index = read_json(dataset_root / "dataset.json")
    scene = next((entry for entry in index.get("scenes", []) if entry.get("scene_id") == scene_id), None)
    if scene is None:
        scene = next((entry for entry in index.get("shards", []) if entry.get("scene_id") == scene_id), None)
    if scene is None:
        raise FileNotFoundError(f"No scene metadata found for {scene_id} in {dataset_root}")
    bbox = [float(value) for value in scene["bbox_mercator"]]
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    latitude, longitude = mercator_to_lat_lon(center_x, center_y)
    size_m = int(round(max(bbox[2] - bbox[0], bbox[3] - bbox[1])))
    return SceneRequest(
        scene_id=scene_id,
        latitude=latitude,
        longitude=longitude,
        size_m=size_m,
        image_px=int(scene.get("image_px", 2048)),
        tile_size_m=int(scene.get("tile_size_m", 25)),
    )


def scene_bbox(scene: SceneRequest) -> tuple[float, float, float, float]:
    center_x, center_y = mercator_from_lat_lon(scene.latitude, scene.longitude)
    half = scene.size_m / 2.0
    return center_x - half, center_y - half, center_x + half, center_y + half


def wms_url(scene: SceneRequest) -> str:
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": "ch.swisstopo.swissimage-product",
            "STYLES": "default",
            "CRS": "EPSG:3857",
            "BBOX": ",".join(str(value) for value in scene_bbox(scene)),
            "WIDTH": scene.image_px,
            "HEIGHT": scene.image_px,
            "FORMAT": "image/jpeg",
        }
    )
    return f"{WMS_BASE_URL}?{query}"


def fetch_url(url: str) -> bytes:
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), context=context, timeout=120) as response:
        return response.read()


def crop_tile(
    image: Image.Image,
    scene: SceneRequest,
    bbox_mercator: tuple[float, float, float, float],
) -> Image.Image:
    left, top, right, bottom = tile_pixel_bounds(scene, bbox_mercator)
    crop_box = (
        max(0, int(round(left))),
        max(0, int(round(top))),
        min(image.width, int(round(right))),
        min(image.height, int(round(bottom))),
    )
    return image.crop(crop_box).convert("RGB")


def tile_pixel_bounds(scene: SceneRequest, bbox_mercator: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    scene_min_x, scene_min_y, scene_max_x, scene_max_y = scene_bbox(scene)
    tile_min_x, tile_min_y, tile_max_x, tile_max_y = bbox_mercator
    scale_x = scene.image_px / (scene_max_x - scene_min_x)
    scale_y = scene.image_px / (scene_max_y - scene_min_y)
    left = (tile_min_x - scene_min_x) * scale_x
    right = (tile_max_x - scene_min_x) * scale_x
    top = (scene_max_y - tile_max_y) * scale_y
    bottom = (scene_max_y - tile_min_y) * scale_y
    return left, top, right, bottom


def split_for_id(tile_id: str) -> str:
    value = int(hashlib.sha1(tile_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "val"
    return "test"


def stratified_splits(candidates: list[Candidate]) -> dict[str, str]:
    by_label: dict[str, list[Candidate]] = {"crosswalk": [], "no_crosswalk": []}
    for candidate in candidates:
        by_label.setdefault(candidate.label, []).append(candidate)

    split_by_tile_id: dict[str, str] = {}
    for label_candidates in by_label.values():
        ordered = sorted(
            label_candidates,
            key=lambda candidate: hashlib.sha1(candidate.tile_id.encode("utf-8")).hexdigest(),
        )
        total = len(ordered)
        if total == 0:
            continue
        train_count = max(1, round(total * 0.8))
        val_count = max(1, round(total * 0.1)) if total >= 3 else 0
        if train_count + val_count >= total and total >= 3:
            train_count = total - val_count - 1
        for index, candidate in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            split_by_tile_id[candidate.tile_id] = split
    return split_by_tile_id


def mercator_from_lat_lon(latitude: float, longitude: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(longitude)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
    return x, y


def mercator_to_lat_lon(x: float, y: float) -> tuple[float, float]:
    longitude = math.degrees(x / EARTH_RADIUS_M)
    latitude = math.degrees(2.0 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2.0)
    return latitude, longitude


def resolve_shard_path(dataset_root: Path, relative_path: str) -> Path:
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise ValueError(f"Unsafe metadata shard path: {relative_path}")
    root = dataset_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Metadata shard path escapes dataset root: {relative_path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_path",
        "split",
        "class_name",
        "source_label",
        "tile_id",
        "scene_id",
        "city",
        "confidence",
        "mask_coverage",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        split = str(row["split"])
        class_name = str(row["class_name"])
        counts.setdefault(split, {"crossing": 0, "no_crossing": 0})
        counts[split][class_name] += 1
    return counts


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)[:128] or "image"


def should_emit_progress(completed: int, total: int) -> bool:
    if completed <= 1 or completed >= total:
        return True
    return completed % max(1, total // 20) == 0


def format_bytes(value: int) -> str:
    if value <= 0:
        return "unknown"
    return f"{value / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    main()
