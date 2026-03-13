"""Tests for src.detection.cat_detector."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.detection.cat_detector import (
    CatDetector,
    Detection,
    DetectionResult,
    CAT_CLASS_ID,
)
from conftest import make_frame, make_random_frame


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

class TestDetection:
    def test_area(self):
        d = Detection(bbox=(10, 20, 110, 170), confidence=0.9)
        assert d.area == 100 * 150

    def test_area_degenerate(self):
        d = Detection(bbox=(50, 50, 50, 50), confidence=0.5)
        assert d.area == 0

    def test_center(self):
        d = Detection(bbox=(0, 0, 100, 200), confidence=0.8)
        assert d.center == (50, 100)

    def test_default_class_id(self):
        d = Detection(bbox=(0, 0, 10, 10), confidence=0.5)
        assert d.class_id == CAT_CLASS_ID


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------

class TestDetectionResult:
    def test_no_cats(self):
        r = DetectionResult()
        assert not r.has_cats
        assert r.cat_count == 0

    def test_with_cats(self):
        dets = [Detection((0, 0, 50, 50), 0.9), Detection((100, 100, 200, 200), 0.8)]
        r = DetectionResult(detections=dets)
        assert r.has_cats
        assert r.cat_count == 2


# ---------------------------------------------------------------------------
# CatDetector (mocked YOLO)
# ---------------------------------------------------------------------------

def _make_mock_yolo_result(boxes_data):
    """
    boxes_data: list of (x1, y1, x2, y2, conf)
    """
    import torch

    mock_result = MagicMock()
    if not boxes_data:
        mock_result.boxes = None
    else:
        boxes = MagicMock()
        box_list = []
        for x1, y1, x2, y2, conf in boxes_data:
            b = MagicMock()
            b.xyxy = [MagicMock()]
            b.xyxy[0].tolist.return_value = [float(x1), float(y1), float(x2), float(y2)]
            b.conf = [MagicMock()]
            b.conf[0].__float__ = lambda self: conf
            # Make float() work on conf mock
            b.conf[0] = conf
            box_list.append(b)
        boxes.__iter__ = lambda s: iter(box_list)
        boxes.__len__ = lambda s: len(box_list)
        mock_result.boxes = boxes

    return [mock_result]


class TestCatDetector:
    def _detector_with_mock(self, boxes_data):
        detector = CatDetector(confidence_threshold=0.4)
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_mock_yolo_result(boxes_data)
        detector._model = mock_model
        return detector

    def test_no_detections(self):
        detector = self._detector_with_mock([])
        frame = make_frame()
        result = detector.detect(frame)
        assert not result.has_cats

    def test_single_cat(self):
        detector = self._detector_with_mock([(10, 20, 100, 150, 0.85)])
        frame = make_frame()
        result = detector.detect(frame)
        assert result.has_cats
        assert result.cat_count == 1
        assert result.detections[0].bbox == (10, 20, 100, 150)
        assert abs(result.detections[0].confidence - 0.85) < 1e-4

    def test_two_cats(self):
        detector = self._detector_with_mock([
            (10, 20, 100, 150, 0.85),
            (200, 300, 400, 450, 0.72),
        ])
        frame = make_frame()
        result = detector.detect(frame)
        assert result.cat_count == 2

    def test_frame_stored_in_result(self):
        detector = self._detector_with_mock([])
        frame = make_frame()
        result = detector.detect(frame)
        assert result.frame is frame

    def test_draw_detections_returns_copy(self):
        detector = self._detector_with_mock([])
        frame = make_frame()
        result = DetectionResult(
            detections=[Detection((50, 50, 200, 200), 0.9)],
            frame=frame,
        )
        drawn = detector.draw_detections(frame, result)
        assert drawn is not frame
        assert drawn.shape == frame.shape

    def test_predict_called_with_cat_class(self):
        detector = self._detector_with_mock([])
        frame = make_frame()
        detector.detect(frame)
        call_kwargs = detector._model.predict.call_args
        assert CAT_CLASS_ID in call_kwargs.kwargs.get("classes", [])

    def test_imgsz_passed_to_predict(self):
        detector = self._detector_with_mock([])
        detector.imgsz = 1280
        frame = make_frame()
        detector.detect(frame)
        call_kwargs = detector._model.predict.call_args
        assert call_kwargs.kwargs.get("imgsz") == 1280

    def test_default_imgsz_is_640(self):
        detector = CatDetector()
        assert detector.imgsz == 640
