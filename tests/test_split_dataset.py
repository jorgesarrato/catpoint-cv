"""Tests for scripts/split_dataset.py"""

import yaml
import shutil
import numpy as np
import pytest
import cv2
from pathlib import Path

from scripts.split_dataset import (
    collect_samples,
    split_samples,
    copy_split,
    write_dataset_yaml,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def labeled_dir(tmp_path):
    """Create a fake labeled dataset with 20 image/label pairs."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(20):
        cv2.imwrite(str(images / f"img_{i:03d}.jpg"), img)
        (labels / f"img_{i:03d}.txt").write_text(
            f"0 0.5 0.5 0.2 0.2\n"
        )
    return tmp_path


# ---------------------------------------------------------------------------
# collect_samples
# ---------------------------------------------------------------------------

class TestCollectSamples:
    def test_finds_all_pairs(self, labeled_dir):
        samples = collect_samples(labeled_dir / "images", labeled_dir / "labels")
        assert len(samples) == 20

    def test_skips_missing_labels(self, labeled_dir):
        (labeled_dir / "labels" / "img_000.txt").unlink()
        samples = collect_samples(labeled_dir / "images", labeled_dir / "labels")
        assert len(samples) == 19

    def test_returns_path_tuples(self, labeled_dir):
        samples = collect_samples(labeled_dir / "images", labeled_dir / "labels")
        img, lbl = samples[0]
        assert img.suffix == ".jpg"
        assert lbl.suffix == ".txt"


# ---------------------------------------------------------------------------
# split_samples
# ---------------------------------------------------------------------------

class TestSplitSamples:
    def test_correct_counts(self, labeled_dir):
        samples = list(range(20))
        train, val, test = split_samples(samples, 0.8, 0.1, seed=42)
        assert len(train) == 16
        assert len(val) == 2
        assert len(test) == 2

    def test_no_overlap(self, labeled_dir):
        samples = list(range(20))
        train, val, test = split_samples(samples, 0.8, 0.1, seed=42)
        all_items = train + val + test
        assert len(all_items) == len(set(all_items))

    def test_all_samples_included(self):
        samples = list(range(20))
        train, val, test = split_samples(samples, 0.8, 0.1, seed=42)
        assert sorted(train + val + test) == sorted(samples)

    def test_reproducible_with_same_seed(self):
        samples = list(range(20))
        a = split_samples(samples, 0.8, 0.1, seed=42)
        b = split_samples(samples, 0.8, 0.1, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        samples = list(range(20))
        a_train, _, _ = split_samples(samples, 0.8, 0.1, seed=0)
        b_train, _, _ = split_samples(samples, 0.8, 0.1, seed=99)
        assert a_train != b_train


# ---------------------------------------------------------------------------
# copy_split
# ---------------------------------------------------------------------------

class TestCopySplit:
    def test_files_copied(self, labeled_dir, tmp_path):
        samples = collect_samples(labeled_dir / "images", labeled_dir / "labels")
        dest = tmp_path / "train"
        copy_split(samples[:5], dest)
        assert len(list((dest / "images").glob("*.jpg"))) == 5
        assert len(list((dest / "labels").glob("*.txt"))) == 5

    def test_originals_preserved(self, labeled_dir, tmp_path):
        samples = collect_samples(labeled_dir / "images", labeled_dir / "labels")
        dest = tmp_path / "train"
        copy_split(samples[:5], dest)
        for img, lbl in samples[:5]:
            assert img.exists()
            assert lbl.exists()


# ---------------------------------------------------------------------------
# write_dataset_yaml
# ---------------------------------------------------------------------------

class TestWriteDatasetYaml:
    def test_yaml_created(self, tmp_path):
        path = write_dataset_yaml(tmp_path, ["salo", "taro"])
        assert path.exists()

    def test_yaml_content(self, tmp_path):
        write_dataset_yaml(tmp_path, ["salo", "taro"])
        cfg = yaml.safe_load((tmp_path / "dataset.yaml").read_text())
        assert cfg["nc"] == 2
        assert cfg["names"] == ["salo", "taro"]
        assert cfg["train"] == "train/images"
        assert cfg["val"] == "val/images"
        assert cfg["test"] == "test/images"

    def test_path_is_absolute(self, tmp_path):
        write_dataset_yaml(tmp_path, ["salo", "taro"])
        cfg = yaml.safe_load((tmp_path / "dataset.yaml").read_text())
        assert Path(cfg["path"]).is_absolute()
