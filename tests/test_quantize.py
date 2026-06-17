import pytest
import torch

from crossing_detector.models import build_crossing_classifier
from crossing_detector.quantize import TraceableCrossingClassifier, choose_quantized_backend


def test_traceable_classifier_matches_regular_resnet_forward():
    model = build_crossing_classifier("fmow-resnet50", pretrained=False)
    model.eval()
    traceable = TraceableCrossingClassifier(model).eval()
    images = torch.randn(2, 3, 64, 64)

    with torch.inference_mode():
        expected = model(images)
        actual = traceable(images)

    assert torch.allclose(actual, expected)


def test_choose_quantized_backend_rejects_unsupported_backend():
    with pytest.raises(ValueError):
        choose_quantized_backend("definitely-not-a-real-backend")
