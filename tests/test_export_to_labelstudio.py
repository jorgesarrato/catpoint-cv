"""Tests for scripts/export_to_labelstudio.py"""

import json
import numpy as np
import pytest
from pathlib import Path

import cv2

from scripts.export_to_labelstudio import (
    load_labeled_filenames,
    bbox_to_percent,
    make_task,
    collect_meta_files,
)


@pytest.fixture
def data_dir(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    cv2.imwrite(str(raw / "img_salo.jpg"), img)
    (raw / "img_salo_meta.json").write_text(json.dumps({
        "full_frame": str(raw / "img_salo.jpg"),
        "cat_count": 1,
        "detections": [{"bbox": [100, 200, 400, 600], "confidence": 0.91}],
    }))

    cv2.imwrite(str(raw / "img_bg.jpg"), img)
    (raw / "img_bg_meta.json").write_text(json.dumps({
        "full_frame": str(raw / "img_bg.jpg"),
        "cat_count": 0,
        "detections": [],
    }))

    return raw, tmp_path


class TestLoadLabeledFilenames:
    def test_returns_empty_if_no_file(self, tmp_path):
        assert load_labeled_filenames(tmp_path / "missing.json") == set()

    def test_extracts_filenames(self, tmp_path):
        export = [
            {"data": {"image": "/data/local-files/?d=data/raw/img1.jpg"}},
            {"data": {"image": "/data/local-files/?d=data/raw/img2.jpg"}},
        ]
        path = tmp_path / "export.json"
        path.write_text(json.dumps(export))
        result = load_labeled_filenames(path)
        assert result == {"img1.jpg", "img2.jpg"}

    def test_handles_http_urls(self, tmp_path):
        export = [{"data": {"image": "http://localhost:8081/data/raw/img1.jpg"}}]
        path = tmp_path / "export.json"
        path.write_text(json.dumps(export))
        result = load_labeled_filenames(path)
        assert "img1.jpg" in result


class TestBboxToPercent:
    def test_full_image(self):
        coords = bbox_to_percent([0, 0, 1920, 1080], 1920, 1080)
        assert coords["x"] == pytest.approx(0.0)
        assert coords["width"] == pytest.approx(100.0)

    def test_offset_box(self):
        coords = bbox_to_percent([192, 108, 384, 216], 1920, 1080)
        assert coords["x"] == pytest.approx(10.0)
        assert coords["y"] == pytest.approx(10.0)


class TestMakeTask:
    def test_cat_task_has_annotations(self, data_dir):
        raw, root = data_dir
        image_path = raw / "img_salo.jpg"
        task = make_task(image_path, [{"bbox": [100, 200, 400, 600], "confidence": 0.91}],
                         document_root=str(root))
        assert "annotations" in task

    def test_background_task_no_annotations(self, data_dir):
        raw, root = data_dir
        image_path = raw / "img_bg.jpg"
        task = make_task(image_path, [], document_root=str(root))
        assert "annotations" not in task

    def test_image_url_format(self, data_dir):
        raw, root = data_dir
        image_path = raw / "img_salo.jpg"
        task = make_task(image_path, [], document_root=str(root))
        assert task["data"]["image"].startswith("/data/local-files/?d=")

    def test_http_base_url(self, data_dir):
        raw, root = data_dir
        image_path = raw / "img_salo.jpg"
        task = make_task(image_path, [], base_url="http://localhost:8081",
                         document_root=str(root))
        assert task["data"]["image"].startswith("http://localhost:8081")


class TestCollectMetaFiles:
    def test_finds_meta_files(self, data_dir):
        raw, _ = data_dir
        files = collect_meta_files(raw)
        assert len(files) == 2

    def test_returns_sorted(self, data_dir):
        raw, _ = data_dir
        files = collect_meta_files(raw)
        assert files == sorted(files)
