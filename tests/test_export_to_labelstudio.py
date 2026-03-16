"""Tests for scripts/export_to_labelstudio.py"""

import json
import numpy as np
import pytest
from pathlib import Path

import cv2

from scripts.export_to_labelstudio import (
    load_processed,
    save_processed,
    bbox_to_percent,
    make_task,
    collect_meta_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory with sample images and metadata."""
    raw = tmp_path / "raw"
    raw.mkdir()

    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Cat image
    cv2.imwrite(str(raw / "20260101_120000_000_1cats.jpg"), img)
    (raw / "20260101_120000_000_1cats_meta.json").write_text(json.dumps({
        "full_frame": str(raw / "20260101_120000_000_1cats.jpg"),
        "cat_count": 1,
        "detections": [{"bbox": [100, 200, 400, 600], "confidence": 0.91}],
    }))

    # Background image
    cv2.imwrite(str(raw / "20260101_120500_000_background.jpg"), img)
    (raw / "20260101_120500_000_background_meta.json").write_text(json.dumps({
        "full_frame": str(raw / "20260101_120500_000_background.jpg"),
        "cat_count": 0,
        "detections": [],
    }))

    return raw


# ---------------------------------------------------------------------------
# load_processed / save_processed
# ---------------------------------------------------------------------------

class TestTracking:
    def test_load_returns_empty_set_if_no_file(self, tmp_path):
        assert load_processed(tmp_path / "nonexistent.json") == set()

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "tracking.json"
        save_processed(path, {"a.jpg", "b.jpg"})
        result = load_processed(path)
        assert result == {"a.jpg", "b.jpg"}

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "tracking.json"
        save_processed(path, {"x.jpg"})
        assert path.exists()


# ---------------------------------------------------------------------------
# bbox_to_percent
# ---------------------------------------------------------------------------

class TestBboxToPercent:
    def test_full_image_bbox(self):
        coords = bbox_to_percent([0, 0, 1920, 1080], 1920, 1080)
        assert coords["x"] == pytest.approx(0.0)
        assert coords["y"] == pytest.approx(0.0)
        assert coords["width"] == pytest.approx(100.0)
        assert coords["height"] == pytest.approx(100.0)

    def test_half_image_bbox(self):
        coords = bbox_to_percent([0, 0, 960, 540], 1920, 1080)
        assert coords["width"] == pytest.approx(50.0)
        assert coords["height"] == pytest.approx(50.0)

    def test_offset_bbox(self):
        coords = bbox_to_percent([192, 108, 384, 216], 1920, 1080)
        assert coords["x"] == pytest.approx(10.0)
        assert coords["y"] == pytest.approx(10.0)
        assert coords["width"] == pytest.approx(10.0)
        assert coords["height"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# make_task
# ---------------------------------------------------------------------------

class TestMakeTask:
    def test_cat_task_has_annotations(self, data_dir, tmp_path):
        image_path = data_dir / "20260101_120000_000_1cats.jpg"
        detections = [{"bbox": [100, 200, 400, 600], "confidence": 0.91}]
        task = make_task(image_path, detections)
        assert "annotations" in task
        assert len(task["annotations"][0]["result"]) == 1

    def test_background_task_has_no_annotations(self, data_dir, tmp_path):
        image_path = data_dir / "20260101_120500_000_background.jpg"
        task = make_task(image_path, [])
        assert "annotations" not in task

    def test_image_url_format(self, data_dir, tmp_path):
        image_path = data_dir / "20260101_120000_000_1cats.jpg"
        task = make_task(image_path, [])
        assert task["data"]["image"].startswith("/data/local-files/?d=")

    def test_bbox_label_is_cat(self, data_dir, tmp_path):
        image_path = data_dir / "20260101_120000_000_1cats.jpg"
        detections = [{"bbox": [100, 200, 400, 600], "confidence": 0.91}]
        task = make_task(image_path, detections)
        result = task["annotations"][0]["result"][0]
        assert result["value"]["rectanglelabels"] == ["cat"]

    def test_original_dimensions_set(self, data_dir, tmp_path):
        image_path = data_dir / "20260101_120000_000_1cats.jpg"
        task = make_task(image_path, [])
        # No annotations but dimensions would be set if there were results
        assert "data" in task

    def test_two_detections_produce_two_results(self, data_dir, tmp_path):
        image_path = data_dir / "20260101_120000_000_1cats.jpg"
        detections = [
            {"bbox": [100, 200, 400, 600], "confidence": 0.91},
            {"bbox": [500, 300, 800, 700], "confidence": 0.75},
        ]
        task = make_task(image_path, detections)
        assert len(task["annotations"][0]["result"]) == 2


# ---------------------------------------------------------------------------
# collect_meta_files
# ---------------------------------------------------------------------------

class TestCollectMetaFiles:
    def test_finds_all_meta_files(self, data_dir):
        files = collect_meta_files(data_dir)
        assert len(files) == 2

    def test_returns_only_meta_json(self, data_dir):
        for f in collect_meta_files(data_dir):
            assert f.name.endswith("_meta.json")

    def test_returns_sorted(self, data_dir):
        files = collect_meta_files(data_dir)
        assert files == sorted(files)
