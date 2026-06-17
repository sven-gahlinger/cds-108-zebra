"""One-batch model interface smoke test."""

from __future__ import annotations

import argparse

import torch

from crossing_detector.models import BACKBONE_SPECS, build_crossing_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="fmow-resnet50", choices=sorted(BACKBONE_SPECS))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=250)
    parser.add_argument("--dropout", type=float, default=0.2)
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrained_group.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_crossing_classifier(
        args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)
    model.eval()

    batch = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    with torch.inference_mode():
        logits = model(batch)

    expected_shape = (args.batch_size, 2)
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}")
    print(
        "ok "
        f"backbone={args.backbone} "
        f"pretrained={args.pretrained} "
        f"input={tuple(batch.shape)} "
        f"logits={tuple(logits.shape)} "
        f"device={device}"
    )


if __name__ == "__main__":
    main()
