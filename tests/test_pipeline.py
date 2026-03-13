"""Tests for src.dataset.pipeline.DatasetPipeline."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.detection.cat_detector import Detection, DetectionResult
from src.detection.preprocessor import CLAHEPreprocessor
from src.dataset.pipeline import DatasetPipeline
from src.dataset.variety_filter import VarietyFilter
from src.dataset.saver import DatasetSaver
from conftest import make_frame


def _make_pipeline(tmp_path, cat_detected: bool, variety_passes: bool):
    """Build a pipeline with mocked detector and real filter/saver."""
    detector = MagicMock()
    det_result = DetectionResult(
        detections=[Detection((10, 10, 200, 200), 0.9)] if cat_detected else [],
        frame=make_frame(),
    )
    detector.detect.return_value = det_result

    variety_filter = MagicMock()
    variety_filter.should_save.return_value = variety_passes

    saver = MagicMock()

    pipeline = DatasetPipeline(detector, variety_filter, saver)
    return pipeline, saver


class TestDatasetPipeline:
    def test_no_cat_returns_none(self, tmp_path):
        pipeline, saver = _make_pipeline(tmp_path, cat_detected=False, variety_passes=True)
        _, result = pipeline.process(make_frame())
        assert result is None
        saver.save.assert_not_called()

    def test_cat_but_no_variety_returns_none(self, tmp_path):
        pipeline, saver = _make_pipeline(tmp_path, cat_detected=True, variety_passes=False)
        _, result = pipeline.process(make_frame())
        assert result is None
        saver.save.assert_not_called()

    def test_cat_and_variety_saves(self, tmp_path):
        pipeline, saver = _make_pipeline(tmp_path, cat_detected=True, variety_passes=True)
        _, result = pipeline.process(make_frame())
        assert result is not None
        saver.save.assert_called_once()

    def test_stats_tracked_correctly(self, tmp_path):
        pipeline, _ = _make_pipeline(tmp_path, cat_detected=True, variety_passes=True)
        frame = make_frame()
        pipeline.process(frame)
        pipeline.process(frame)
        stats = pipeline.stats
        assert stats["total_frames"] == 2
        assert stats["detection_frames"] == 2
        assert stats["saved_frames"] == 2

    def test_no_cat_detection_frame_not_counted(self, tmp_path):
        pipeline, _ = _make_pipeline(tmp_path, cat_detected=False, variety_passes=False)
        pipeline.process(make_frame())
        stats = pipeline.stats
        assert stats["total_frames"] == 1
        assert stats["detection_frames"] == 0
        assert stats["saved_frames"] == 0

    def test_variety_filter_not_called_without_cat(self, tmp_path):
        """Variety filter should only run when cats are detected (efficiency)."""
        pipeline, _ = _make_pipeline(tmp_path, cat_detected=False, variety_passes=True)
        pipeline.process(make_frame())
        pipeline.variety_filter.should_save.assert_not_called()

    def test_process_returns_preprocessed_frame(self, tmp_path):
        """process() must always return the preprocessed frame as first element."""
        pipeline, _ = _make_pipeline(tmp_path, cat_detected=False, variety_passes=False)
        frame = make_frame()
        preprocessed, _ = pipeline.process(frame)
        assert preprocessed is not None
        assert isinstance(preprocessed, np.ndarray)

    def test_integration_with_real_components(self, tmp_path):
        """Integration: real filter and saver, mocked detector."""
        detector = MagicMock()
        frame = make_frame(color=(50, 100, 150))
        det_result = DetectionResult(
            detections=[Detection((20, 30, 400, 400), 0.88)],
            frame=frame,
        )
        detector.detect.return_value = det_result

        pipeline = DatasetPipeline(
            detector=detector,
            variety_filter=VarietyFilter(min_interval_sec=0.0, max_interval_sec=9999),
            saver=DatasetSaver(output_dir=str(tmp_path)),
        )

        _, result = pipeline.process(frame)
        assert result is not None
        assert pipeline.stats["saved_frames"] == 1
        saved_files = list(tmp_path.iterdir())
        assert len(saved_files) > 0

    def test_preprocessor_is_called(self, tmp_path):
        """Pipeline must call preprocessor.process() on every frame."""
        detector = MagicMock()
        detector.detect.return_value = DetectionResult(detections=[], frame=make_frame())

        preprocessor = MagicMock(spec=CLAHEPreprocessor)
        preprocessor.process.return_value = make_frame()

        pipeline = DatasetPipeline(
            detector=detector,
            variety_filter=MagicMock(**{"should_save.return_value": False}),
            saver=MagicMock(),
            preprocessor=preprocessor,
        )

        pipeline.process(make_frame())
        preprocessor.process.assert_called_once()

    def test_preprocessor_disabled_passthrough(self, tmp_path):
        """With CLAHE disabled the frame reaching the detector should be unchanged."""
        received = {}

        def capture_detect(frame):
            received["frame"] = frame
            return DetectionResult(detections=[], frame=frame)

        detector = MagicMock()
        detector.detect.side_effect = capture_detect

        original = make_frame(color=(77, 88, 99))
        pipeline = DatasetPipeline(
            detector=detector,
            variety_filter=MagicMock(**{"should_save.return_value": False}),
            saver=MagicMock(),
            preprocessor=CLAHEPreprocessor(enabled=False),
        )

        pipeline.process(original)
        assert received["frame"] is original
