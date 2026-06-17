import torch

from crossing_detector.models import build_crossing_classifier


def test_model_outputs_two_class_logits_per_image_without_downloading_weights():
    model = build_crossing_classifier("fmow-resnet50", pretrained=False)
    model.eval()
    images = torch.randn(2, 3, 64, 64)

    with torch.inference_mode():
        logits = model(images)

    assert logits.shape == (2, 2)
