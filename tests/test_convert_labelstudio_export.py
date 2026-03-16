"""Tests for scripts/convert_labelstudio_export.py"""

import json
import numpy as np
import pytest
import cv2
from pathlib import Path

from scripts.convert_labelstudio_export import (
    parse_image_path,
    percent_to_yolo,
    convert_task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_dir(tmp_path):
    """Create fake raw images."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.imwrite(str(raw / "img_salo.jpg"), img)
    cv2.imwrite(str(raw / "img_taro.jpg"), img)
    cv2.imwrite(str(raw / "img_bg.jpg"), img)
    return raw, tmp_path


@pytest.fixture
def output_dirs(tmp_path):
    images = tmp_path / "labeled" / "images"
    labels = tmp_path / "labeled" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    return images, labels


def make_task(image_name: str, boxes: list, document_root: str) -> dict:
    return {
        "data": {"image": f"/data/local-files/?d=data/raw/{image_name}"},
        "annotations": [{"result": [
            {
                "type": "rectanglelabels",
                "value": {
                    "x": b["x"], "y": b["y"],
                    "width": b["w"], "height": b["h"],
                    "rectanglelabels": [b["label"]],
                },
                "original_width": 1920,
                "original_height": 1080,
            }
            for b in boxes
        ]}] if boxes else []
    }


# ---------------------------------------------------------------------------
# parse_image_path
# ---------------------------------------------------------------------------

class TestParseImagePath:
    def test_local_files_url(self, tmp_path):
        path = parse_image_path(
            "/data/local-files/?d=data/raw/img.jpg",
            document_root=str(tmp_path)
        )
        assert path == tmp_path / "data" / "raw" / "img.jpg"

    def test_http_url(self, tmp_path):
        path = parse_image_path(
            "http://localhost:8081/data/raw/img.jpg",
            document_root=str(tmp_path)
        )
        assert path == tmp_path / "data" / "raw" / "img.jpg"

    def test_nested_path(self, tmp_path):
        path = parse_image_path(
            "/data/local-files/?d=data/raw/subdir/img.jpg",
            document_root=str(tmp_path)
        )
        assert path == tmp_path / "data" / "raw" / "subdir" / "img.jpg"


# ---------------------------------------------------------------------------
# percent_to_yolo
# ---------------------------------------------------------------------------

class TestPercentToYolo:
    def test_centre_box(self):
        cx, cy, w, h = percent_to_yolo(25.0, 25.0, 50.0, 50.0)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)

    def test_top_left_box(self):
        cx, cy, w, h = percent_to_yolo(0.0, 0.0, 20.0, 20.0)
        assert cx == pytest.approx(0.1)
        assert cy == pytest.approx(0.1)
        assert w == pytest.approx(0.2)
        assert h == pytest.approx(0.2)

    def test_full_image_box(self):
        cx, cy, w, h = percent_to_yolo(0.0, 0.0, 100.0, 100.0)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(1.0)

    def test_values_normalised(self):
        cx, cy, w, h = percent_to_yolo(10.0, 10.0, 30.0, 40.0)
        for v in [cx, cy, w, h]:
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# convert_task
# ---------------------------------------------------------------------------

class TestConvertTask:
    def test_cat_image_creates_label(self, raw_dir, output_dirs):
        raw, root = raw_dir
        images_dir, labels_dir = output_dirs
        task = make_task("img_salo.jpg", [
            {"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0, "label": "salo"}
        ], str(root))
        n, warnings = convert_task(task, images_dir, labels_dir, ["salo", "taro"], str(root))
        assert n == 1
        assert (labels_dir / "img_salo.txt").exists()
        assert warnings == []

    def test_label_file_content(self, raw_dir, output_dirs):
        raw, root = raw_dir
        images_dir, labels_dir = output_dirs
        task = make_task("img_salo.jpg", [
            {"x": 25.0, "y": 25.0, "w": 50.0, "h": 50.0, "label": "salo"}
        ], str(root))
        convert_task(task, images_dir, labels_dir, ["salo", "taro"], str(root))
        lines = (labels_dir / "img_salo.txt").read_text().strip().split("\n")
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0"  # salo = class 0
        assert float(parts[1]) == pytest.approx(0.5)  # cx
        assert float(parts[2]) == pytest.approx(0.5)  # cy

    def test_taro_gets_class_id_1(self, raw_dir, output_dirs):
        raw, root = raw_dir
        images_dir, labels_dir = output_dirs
        task = make_task("img_taro.jpg", [
            {"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0, "label": "taro"}
        ], str(root))
        convert_task(task, images_dir, labels_dir, ["salo", "taro"], str(root))
        line = (labels_dir / "img_taro.txt").read_text().strip().split("\n")[0]
        assert line.split()[0] == "1"

    def test_background_creates_empty_label(self, raw_dir, output_dirs):
        raw, root = raw_dir
        images_dir, labels_dir = output_dirs
        task = make_task("img_bg.jpg", [], str(root))
        n, warnings = convert_task(task, images_dir, labels_dir, ["salo", "taro"], str(root))
        assert n == 0
        assert (labels_dir / "img_bg.txt").read_text() == ""

    def test_image_copied(self, raw_dir, output_dirs):
        raw, root = raw_dir
        images_dir, labels_dir = output_dirs
        task = make_task("img_salo.jpg", [
            {"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0, "label": "salo"}
        ], str(root))
        convert_task(task, images_dir, labels_dir, ["salo", "taro"], str(root))
        assert (images_dir / "img_salo.jpg").exists()

    def test_unknown_label_skipped_with_warning(self, raw_dir, output_dirs):
        raw, root = raw_dir
        images_dir, labels_dir = output_dirs
        task = make_task("img_salo.jpg", [
            {"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0, "label": "unknown_cat"}
        ], str(root))
        n, warnings = convert_task(task, images_dir, labels_dir, ["salo", "taro"], str(root))
        assert n == 0
        assert any("unknown label" in w for w in warnings)

    def test_missing_image_returns_warning(self, raw_dir, output_dirs):
        raw, root = raw_dir
        images_dir, labels_dir = output_dirs
        task = make_task("nonexistent.jpg", [
            {"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0, "label": "salo"}
        ], str(root))
        n, warnings = convert_task(task, images_dir, labels_dir, ["salo", "taro"], str(root))
        assert n == 0
        assert any("not found" in w for w in warnings)

    def test_two_boxes_same_image(self, raw_dir, output_dirs):
        raw, root = raw_dir
        images_dir, labels_dir = output_dirs
        task = make_task("img_salo.jpg", [
            {"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0, "label": "salo"},
            {"x": 50.0, "y": 50.0, "w": 20.0, "h": 20.0, "label": "taro"},
        ], str(root))
        n, _ = convert_task(task, images_dir, labels_dir, ["salo", "taro"], str(root))
        assert n == 2
        lines = (labels_dir / "img_salo.txt").read_text().strip().split("\n")
        assert len(lines) == 2
