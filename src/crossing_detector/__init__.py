"""SwissImage pedestrian-crossing classification helpers."""

from crossing_detector.models import (
    BACKBONE_SPECS,
    SwissImageCrossingClassifier,
    build_crossing_classifier,
)

__all__ = [
    "BACKBONE_SPECS",
    "SwissImageCrossingClassifier",
    "build_crossing_classifier",
]
