"""
Integration test: verifies that the real YOLOv8n model can detect a cat
(COCO class 15) in a sample image downloaded from the web.

This test requires:
  - ultralytics installed
  - internet access on first run (to download yolov8n.pt weights)

Run with: pytest tests/test_yolo_integration.py -v
"""

import urllib.request
import numpy as np
import cv2
import pytest

from src.detection.cat_detector import CatDetector, CAT_CLASS_ID


# Public domain cat images — tried in order until one succeeds
CAT_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
    "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?w=320",
    "https://placecats.com/320/240",
]


def _download_image(urls) -> np.ndarray:
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (cat-tracker-test/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        except Exception:
            continue
    pytest.skip("Could not download a cat test image from any source (no network?)")


@pytest.fixture(scope="module")
def cat_image() -> np.ndarray:
    """Download a sample cat image once per test session."""
    return _download_image(CAT_IMAGE_URLS)


@pytest.fixture(scope="module")
def detector() -> CatDetector:
    """Real detector with default weights — loaded once per session."""
    return CatDetector(model_path="yolov8n.pt", confidence_threshold=0.3)


class TestYoloRealCatDetection:
    def test_cat_detected(self, cat_image, detector):
        """Model must detect at least one cat in the sample image."""
        result = detector.detect(cat_image)
        assert result.has_cats, (
            "YOLOv8n failed to detect any cat (class 15) in the sample image. "
            "Check model weights or lower the confidence threshold."
        )

    def test_detected_class_is_cat(self, cat_image, detector):
        """All returned detections must have class_id == CAT_CLASS_ID (15)."""
        result = detector.detect(cat_image)
        for det in result.detections:
            assert det.class_id == CAT_CLASS_ID, (
                f"Expected class_id {CAT_CLASS_ID}, got {det.class_id}"
            )

    def test_confidence_above_threshold(self, cat_image, detector):
        """Every detection must meet the detector's own confidence threshold."""
        result = detector.detect(cat_image)
        for det in result.detections:
            assert det.confidence >= detector.confidence_threshold, (
                f"Detection confidence {det.confidence:.3f} is below "
                f"threshold {detector.confidence_threshold}"
            )

    def test_bbox_within_image_bounds(self, cat_image, detector):
        """Bounding boxes must lie within the image dimensions."""
        h, w = cat_image.shape[:2]
        result = detector.detect(cat_image)
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            assert 0 <= x1 < x2 <= w, f"Invalid x coords: {x1}, {x2} (width={w})"
            assert 0 <= y1 < y2 <= h, f"Invalid y coords: {y1}, {y2} (height={h})"

    def test_detect_all_includes_cat(self, cat_image, detector):
        """detect_all() without class filter must also find 'cat' in the image."""
        found = detector.detect_all(cat_image)
        assert "cat" in found, (
            f"detect_all() did not find 'cat' in results. Found: {list(found.keys())}"
        )
