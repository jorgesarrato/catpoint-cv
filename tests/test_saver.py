"""Tests for src.dataset.saver.DatasetSaver."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.detection.cat_detector import Detection, DetectionResult
from src.dataset.saver import DatasetSaver
from conftest import make_frame


@pytest.fixture
def tmp_saver(tmp_path):
    return DatasetSaver(output_dir=str(tmp_path), save_crops=True, jpg_quality=85)


@pytest.fixture
def single_cat_result():
    frame = make_frame(480, 640, color=(80, 120, 200))
    det = Detection(bbox=(50, 60, 300, 400), confidence=0.91)
    return DetectionResult(detections=[det], frame=frame)


@pytest.fixture
def two_cat_result():
    frame = make_frame(480, 640, color=(30, 40, 50))
    dets = [
        Detection(bbox=(10, 10, 200, 200), confidence=0.88),
        Detection(bbox=(300, 200, 600, 450), confidence=0.75),
    ]
    return DetectionResult(detections=dets, frame=frame)


class TestDatasetSaver:
    def test_returns_stem_on_save(self, tmp_saver, single_cat_result):
        stem = tmp_saver.save(single_cat_result)
        assert stem is not None
        assert "1cats" in stem

    def test_full_frame_saved(self, tmp_saver, single_cat_result):
        stem = tmp_saver.save(single_cat_result)
        out = Path(tmp_saver.output_dir) / f"{stem}.jpg"
        assert out.exists()

    def test_crop_saved_for_single_cat(self, tmp_saver, single_cat_result):
        stem = tmp_saver.save(single_cat_result)
        crop = Path(tmp_saver.output_dir) / f"{stem}_crop_0.jpg"
        assert crop.exists()

    def test_two_crops_saved(self, tmp_saver, two_cat_result):
        stem = tmp_saver.save(two_cat_result)
        assert (Path(tmp_saver.output_dir) / f"{stem}_crop_0.jpg").exists()
        assert (Path(tmp_saver.output_dir) / f"{stem}_crop_1.jpg").exists()

    def test_metadata_json_saved(self, tmp_saver, single_cat_result):
        stem = tmp_saver.save(single_cat_result)
        meta_path = Path(tmp_saver.output_dir) / f"{stem}_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["cat_count"] == 1
        assert len(meta["detections"]) == 1
        assert meta["detections"][0]["bbox"] == [50, 60, 300, 400]

    def test_metadata_confidence(self, tmp_saver, single_cat_result):
        stem = tmp_saver.save(single_cat_result)
        meta = json.loads(
            (Path(tmp_saver.output_dir) / f"{stem}_meta.json").read_text()
        )
        assert abs(meta["detections"][0]["confidence"] - 0.91) < 0.001

    def test_no_save_without_cats(self, tmp_saver):
        frame = make_frame()
        result = DetectionResult(detections=[], frame=frame)
        stem = tmp_saver.save(result)
        assert stem is None

    def test_no_save_without_frame(self, tmp_saver):
        det = Detection(bbox=(0, 0, 100, 100), confidence=0.9)
        result = DetectionResult(detections=[det], frame=None)
        stem = tmp_saver.save(result)
        assert stem is None

    def test_session_count_increments(self, tmp_saver, single_cat_result, two_cat_result):
        assert tmp_saver.session_count == 0
        tmp_saver.save(single_cat_result)
        assert tmp_saver.session_count == 1
        tmp_saver.save(two_cat_result)
        assert tmp_saver.session_count == 2

    def test_small_crop_skipped(self, tmp_saver):
        """Crops smaller than min_crop_px should not be saved."""
        frame = make_frame()
        det = Detection(bbox=(10, 10, 20, 20), confidence=0.9)  # only 10x10
        result = DetectionResult(detections=[det], frame=frame)
        stem = tmp_saver.save(result)
        crop = Path(tmp_saver.output_dir) / f"{stem}_crop_0.jpg"
        assert not crop.exists()

    def test_save_crops_false(self, tmp_path, single_cat_result):
        saver = DatasetSaver(output_dir=str(tmp_path), save_crops=False)
        stem = saver.save(single_cat_result)
        crop = Path(tmp_path) / f"{stem}_crop_0.jpg"
        assert not crop.exists()

    def test_external_frame_used_over_result_frame(self, tmp_saver):
        internal_frame = make_frame(color=(0, 0, 0))
        external_frame = make_frame(color=(255, 0, 0))
        det = Detection(bbox=(10, 10, 300, 300), confidence=0.9)
        result = DetectionResult(detections=[det], frame=internal_frame)
        stem = tmp_saver.save(result, frame=external_frame)
        assert stem is not None  # just ensure it ran cleanly


class TestDatasetSaverBackground:
    def test_background_returns_stem(self, tmp_saver):
        stem = tmp_saver.save_background(make_frame())
        assert stem is not None
        assert "background" in stem

    def test_background_frame_saved(self, tmp_saver):
        stem = tmp_saver.save_background(make_frame())
        assert (Path(tmp_saver.output_dir) / f"{stem}.jpg").exists()

    def test_background_meta_saved(self, tmp_saver):
        stem = tmp_saver.save_background(make_frame())
        meta_path = Path(tmp_saver.output_dir) / f"{stem}_meta.json"
        assert meta_path.exists()
        import json
        meta = json.loads(meta_path.read_text())
        assert meta["cat_count"] == 0
        assert meta["detections"] == []
        assert meta["crops"] == []

    def test_background_no_crops_saved(self, tmp_saver):
        stem = tmp_saver.save_background(make_frame())
        crops = list(Path(tmp_saver.output_dir).glob(f"{stem}_crop_*.jpg"))
        assert crops == []

    def test_background_none_frame_returns_none(self, tmp_saver):
        assert tmp_saver.save_background(None) is None
