"""Tests for scripts/merge_labelstudio_exports.py"""

import json
import pytest
from pathlib import Path

from scripts.merge_labelstudio_exports import (
    extract_image_filename,
    load_export,
    merge_exports,
)


def make_task(filename: str, label: str = "salo") -> dict:
    return {
        "data": {"image": f"/data/local-files/?d=data/raw/{filename}"},
        "annotations": [{"result": [{"value": {"rectanglelabels": [label]}}]}],
    }


@pytest.fixture
def export_dir(tmp_path):
    a = [make_task("img1.jpg"), make_task("img2.jpg")]
    b = [make_task("img2.jpg"), make_task("img3.jpg")]  # img2 is a duplicate
    (tmp_path / "export_a.json").write_text(json.dumps(a))
    (tmp_path / "export_b.json").write_text(json.dumps(b))
    return tmp_path


class TestExtractImageFilename:
    def test_local_files_url(self):
        task = make_task("img1.jpg")
        assert extract_image_filename(task) == "img1.jpg"

    def test_http_url(self):
        task = {"data": {"image": "http://localhost:8081/data/raw/img1.jpg"}}
        assert extract_image_filename(task) == "img1.jpg"

    def test_missing_image_returns_none(self):
        assert extract_image_filename({}) is None

    def test_nested_path(self):
        task = {"data": {"image": "/data/local-files/?d=data/raw/subdir/img1.jpg"}}
        assert extract_image_filename(task) == "img1.jpg"


class TestLoadExport:
    def test_loads_list(self, tmp_path):
        tasks = [make_task("a.jpg"), make_task("b.jpg")]
        path = tmp_path / "export.json"
        path.write_text(json.dumps(tasks))
        result = load_export(path)
        assert len(result) == 2

    def test_loads_wrapped_dict(self, tmp_path):
        tasks = [make_task("a.jpg")]
        path = tmp_path / "export.json"
        path.write_text(json.dumps({"tasks": tasks}))
        result = load_export(path)
        assert len(result) == 1

    def test_bad_file_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{")
        result = load_export(path)
        assert result == []


class TestMergeExports:
    def test_deduplicates(self, export_dir):
        files = sorted(export_dir.glob("*.json"))
        merged, stats = merge_exports(files)
        assert len(merged) == 3  # img1, img2, img3
        assert stats["duplicates"] == 1

    def test_all_files_counted(self, export_dir):
        files = sorted(export_dir.glob("*.json"))
        _, stats = merge_exports(files)
        assert stats["files"] == 2
        assert stats["total"] == 4

    def test_single_file(self, tmp_path):
        tasks = [make_task("a.jpg"), make_task("b.jpg")]
        path = tmp_path / "export.json"
        path.write_text(json.dumps(tasks))
        merged, stats = merge_exports([path])
        assert len(merged) == 2
        assert stats["duplicates"] == 0

    def test_preserves_annotations(self, export_dir):
        files = sorted(export_dir.glob("*.json"))
        merged, _ = merge_exports(files)
        filenames = {extract_image_filename(t) for t in merged}
        assert filenames == {"img1.jpg", "img2.jpg", "img3.jpg"}
