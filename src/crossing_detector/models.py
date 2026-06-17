"""Backbone and classifier definitions for SwissImage crossing detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BackboneSpec:
    """Definition for one supported TorchGeo/timm backbone."""

    timm_name: str
    torchgeo_weight_class: str
    torchgeo_weight_name: str
    in_chans: int
    description: str


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "fmow-resnet50": BackboneSpec(
        timm_name="resnet50",
        torchgeo_weight_class="ResNet50_Weights",
        torchgeo_weight_name="FMOW_RGB_GASSL",
        in_chans=3,
        description="Recommended CNN: ResNet-50 pretrained on fMoW RGB remote-sensing imagery.",
    ),
    "sentinel2-resnet18-rgb": BackboneSpec(
        timm_name="resnet18",
        torchgeo_weight_class="ResNet18_Weights",
        torchgeo_weight_name="SENTINEL2_RGB_MOCO",
        in_chans=3,
        description="Fallback CNN: ResNet-18 pretrained on Sentinel-2 RGB imagery.",
    ),
    "sentinel2-resnet50-rgb": BackboneSpec(
        timm_name="resnet50",
        torchgeo_weight_class="ResNet50_Weights",
        torchgeo_weight_name="SENTINEL2_RGB_MOCO",
        in_chans=3,
        description="Fallback CNN: ResNet-50 pretrained on Sentinel-2 RGB imagery.",
    ),
}


def _import_timm():
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - exercised only before install
        raise RuntimeError(
            "Missing dependency 'timm'. Install the project with `pip install -e .`."
        ) from exc
    return timm


def _resolve_torchgeo_weights(spec: BackboneSpec):
    try:
        import torchgeo.models as torchgeo_models
    except ImportError as exc:  # pragma: no cover - exercised only before install
        raise RuntimeError(
            "Missing dependency 'torchgeo'. Install the project with `pip install -e .`."
        ) from exc

    try:
        weight_class = getattr(torchgeo_models, spec.torchgeo_weight_class)
        return getattr(weight_class, spec.torchgeo_weight_name)
    except AttributeError as exc:
        raise RuntimeError(
            "The installed TorchGeo version does not expose "
            f"{spec.torchgeo_weight_class}.{spec.torchgeo_weight_name}."
        ) from exc


def build_timm_feature_backbone(
    backbone_id: str = "fmow-resnet50",
    *,
    pretrained: bool = True,
    progress: bool = True,
) -> nn.Module:
    """Create a classifier-free timm backbone and optionally load TorchGeo weights."""

    if backbone_id not in BACKBONE_SPECS:
        supported = ", ".join(sorted(BACKBONE_SPECS))
        raise ValueError(f"Unknown backbone '{backbone_id}'. Supported: {supported}.")

    spec = BACKBONE_SPECS[backbone_id]
    timm = _import_timm()
    backbone = timm.create_model(
        spec.timm_name,
        pretrained=False,
        in_chans=spec.in_chans,
        num_classes=0,
    )

    if pretrained:
        weights = _resolve_torchgeo_weights(spec)
        state_dict = weights.get_state_dict(progress=progress)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        allowed_missing = {"fc.weight", "fc.bias", "classifier.weight", "classifier.bias"}
        meaningful_missing = [key for key in missing if key not in allowed_missing]
        if unexpected or meaningful_missing:
            raise RuntimeError(
                "TorchGeo weights did not cleanly load into the timm backbone. "
                f"Missing={meaningful_missing}, unexpected={unexpected}"
            )

    return backbone


def build_aggressive_mlp_head(
    input_dim: int,
    *,
    output_dim: int = 2,
    shrink_factor: int = 8,
    min_hidden_dim: int = 64,
    dropout: float = 0.2,
) -> nn.Sequential:
    """Build the requested MLP: same-size hidden layer, then aggressive taper."""

    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if shrink_factor < 2:
        raise ValueError("shrink_factor must be at least 2.")
    if min_hidden_dim < output_dim:
        raise ValueError("min_hidden_dim must be greater than or equal to output_dim.")

    dims = [input_dim, input_dim]
    current_dim = input_dim
    while current_dim > min_hidden_dim:
        current_dim = current_dim // shrink_factor
        dims.append(current_dim)
    dims.append(output_dim)

    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != output_dim:
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class SwissImageCrossingClassifier(nn.Module):
    """Two-class classifier with a removable/pretrained CNN feature backbone."""

    def __init__(self, backbone: nn.Module, *, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = self._feature_dim(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = build_aggressive_mlp_head(
            self.feature_dim,
            output_dim=2,
            shrink_factor=8,
            min_hidden_dim=64,
            dropout=dropout,
        )

    @staticmethod
    def _feature_dim(backbone: nn.Module) -> int:
        feature_dim = getattr(backbone, "num_features", None)
        if feature_dim is None:
            feature_dim = getattr(backbone, "feature_info", None)
            if feature_dim is not None:
                feature_dim = feature_dim.channels()[-1]
        if not isinstance(feature_dim, int):
            raise ValueError(
                "Could not infer backbone feature dimension. Use a timm backbone "
                "with `num_features`, or extend SwissImageCrossingClassifier."
            )
        return feature_dim

    def forward(self, images: Tensor) -> Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(images)
        else:
            features = self.backbone(images)

        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim == 4:
            features = self.pool(features).flatten(1)
        elif features.ndim > 2:
            features = features.flatten(1)

        return self.head(features)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def unfreeze_last_stage(self) -> None:
        self.freeze_backbone()
        last_stage = getattr(self.backbone, "layer4", None)
        if last_stage is None:
            raise RuntimeError("The selected backbone has no `layer4` stage to unfreeze.")
        for parameter in last_stage.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True


def build_crossing_classifier(
    backbone_id: str = "fmow-resnet50",
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
    progress: bool = True,
) -> SwissImageCrossingClassifier:
    backbone = build_timm_feature_backbone(
        backbone_id=backbone_id,
        pretrained=pretrained,
        progress=progress,
    )
    return SwissImageCrossingClassifier(backbone, dropout=dropout)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def set_frozen_batchnorm_eval(module: nn.Module) -> None:
    """Keep frozen BatchNorm layers from updating running statistics."""

    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in child.parameters()):
                child.eval()
