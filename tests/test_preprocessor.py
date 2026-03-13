"""Tests for src.detection.preprocessor.CLAHEPreprocessor."""

import numpy as np
import pytest
import cv2

from src.detection.preprocessor import CLAHEPreprocessor
from conftest import make_frame, make_random_frame


class TestCLAHEPreprocessor:
    def test_output_same_shape(self):
        pre = CLAHEPreprocessor(enabled=True)
        frame = make_random_frame()
        out = pre.process(frame)
        assert out.shape == frame.shape

    def test_output_same_dtype(self):
        pre = CLAHEPreprocessor(enabled=True)
        frame = make_random_frame()
        out = pre.process(frame)
        assert out.dtype == frame.dtype

    def test_input_not_modified(self):
        pre = CLAHEPreprocessor(enabled=True)
        frame = make_random_frame(seed=1)
        original = frame.copy()
        pre.process(frame)
        np.testing.assert_array_equal(frame, original)

    def test_disabled_returns_same_frame(self):
        pre = CLAHEPreprocessor(enabled=False)
        frame = make_random_frame()
        out = pre.process(frame)
        assert out is frame  # exact same object, no copy

    def test_overexposed_frame_has_lower_mean_after_clahe(self):
        """CLAHE should reduce the mean brightness of an overexposed frame."""
        pre = CLAHEPreprocessor(clip_limit=3.0, enabled=True)
        # Create a heavily overexposed frame
        frame = np.full((480, 640, 3), 240, dtype=np.uint8)
        # Add slight variation so CLAHE has something to work with
        rng = np.random.default_rng(0)
        frame = np.clip(frame + rng.integers(-15, 15, frame.shape), 0, 255).astype(np.uint8)
        out = pre.process(frame)
        assert out.mean() < frame.mean(), (
            "CLAHE should reduce mean brightness of an overexposed frame"
        )

    def test_dark_frame_has_higher_mean_after_clahe(self):
        """CLAHE should increase the mean brightness of an underexposed frame."""
        pre = CLAHEPreprocessor(clip_limit=3.0, enabled=True)
        frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        rng = np.random.default_rng(0)
        frame = np.clip(frame + rng.integers(-10, 10, frame.shape), 0, 255).astype(np.uint8)
        out = pre.process(frame)
        assert out.mean() > frame.mean(), (
            "CLAHE should increase mean brightness of an underexposed frame"
        )

    def test_clip_limit_affects_output(self):
        """Different clip limits should produce different outputs."""
        frame = make_random_frame(seed=5)
        out_low = CLAHEPreprocessor(clip_limit=1.0, enabled=True).process(frame)
        out_high = CLAHEPreprocessor(clip_limit=8.0, enabled=True).process(frame)
        assert not np.array_equal(out_low, out_high)

    def test_normal_frame_unchanged_approximately(self):
        """A well-exposed frame should not be drastically altered."""
        pre = CLAHEPreprocessor(clip_limit=2.0, enabled=True)
        frame = make_random_frame(seed=3)
        out = pre.process(frame)
        # Mean should not shift by more than 30 points on a random well-exposed frame
        assert abs(float(out.mean()) - float(frame.mean())) < 30
