"""Tests for src.dataset.variety_filter.VarietyFilter."""

import time
import numpy as np
import pytest

from src.dataset.variety_filter import VarietyFilter
from conftest import make_frame, make_random_frame


class TestVarietyFilter:
    def test_first_frame_always_saved(self):
        vf = VarietyFilter(min_interval_sec=0.0)
        frame = make_frame()
        assert vf.should_save(frame) is True

    def test_identical_frame_not_saved_immediately(self):
        vf = VarietyFilter(min_interval_sec=0.0, similarity_threshold=0.1, max_interval_sec=9999)
        frame = make_frame(color=(100, 100, 100))
        vf.should_save(frame)  # accept first
        assert vf.should_save(frame) is False  # identical → skip

    def test_different_frame_saved(self):
        vf = VarietyFilter(min_interval_sec=0.0, similarity_threshold=0.05, max_interval_sec=9999)
        # Use frames with distinct hues — pure black/white have no saturation
        # so their HSV histograms are indistinguishable
        frame_a = make_frame(color=(200, 50, 50))   # blue-ish
        frame_b = make_frame(color=(50, 50, 200))   # red-ish
        vf.should_save(frame_a)
        assert vf.should_save(frame_b) is True

    def test_min_interval_respected(self):
        vf = VarietyFilter(min_interval_sec=60.0, similarity_threshold=0.0, max_interval_sec=9999)
        frame_a = make_frame(color=(0, 0, 0))
        frame_b = make_frame(color=(255, 255, 255))
        vf.should_save(frame_a)
        # Even a very different frame should be rejected within min_interval
        assert vf.should_save(frame_b) is False

    def test_max_interval_forces_save(self, monkeypatch):
        """Even an identical frame is saved once max_interval elapses."""
        vf = VarietyFilter(
            min_interval_sec=0.0,
            similarity_threshold=0.99,  # very strict — almost nothing passes
            max_interval_sec=5.0,
        )
        frame = make_frame(color=(128, 128, 128))
        vf.should_save(frame)  # first save

        # Simulate time passing beyond max_interval
        vf._last_save_time -= 6.0
        assert vf.should_save(frame) is True

    def test_reset_clears_state(self):
        vf = VarietyFilter(min_interval_sec=0.0, similarity_threshold=0.01, max_interval_sec=9999)
        frame = make_frame()
        vf.should_save(frame)
        vf.reset()
        # After reset, should behave like a fresh filter
        assert vf.should_save(frame) is True

    def test_random_vs_random_different_seeds(self):
        """Frames from different seeds should pass the variety filter."""
        vf = VarietyFilter(min_interval_sec=0.0, similarity_threshold=0.05, max_interval_sec=9999)
        frame_a = make_random_frame(seed=0)
        frame_b = make_random_frame(seed=42)
        vf.should_save(frame_a)
        assert vf.should_save(frame_b) is True

    def test_random_vs_itself(self):
        """Same random frame twice should not pass (below threshold)."""
        vf = VarietyFilter(min_interval_sec=0.0, similarity_threshold=0.1, max_interval_sec=9999)
        frame = make_random_frame(seed=7)
        vf.should_save(frame)
        assert vf.should_save(frame) is False


class TestVarietyFilterBackground:
    def test_first_background_always_saved(self):
        vf = VarietyFilter(background_interval_sec=60.0)
        assert vf.should_save_background() is True

    def test_background_not_saved_within_interval(self):
        vf = VarietyFilter(background_interval_sec=60.0)
        vf.should_save_background()
        assert vf.should_save_background() is False

    def test_background_saved_after_interval(self):
        vf = VarietyFilter(background_interval_sec=60.0)
        vf.should_save_background()
        vf._last_background_save_time -= 61.0
        assert vf.should_save_background() is True

    def test_background_independent_from_cat_filter(self):
        """Saving a cat frame does not affect background timer and vice versa."""
        vf = VarietyFilter(min_interval_sec=0.0, background_interval_sec=60.0)
        frame = make_frame()
        vf.should_save(frame)
        # Background timer untouched — first call should still pass
        assert vf.should_save_background() is True

    def test_reset_clears_background_timer(self):
        vf = VarietyFilter(background_interval_sec=60.0)
        vf.should_save_background()
        vf.reset()
        assert vf.should_save_background() is True

    def test_reset_background_timer_delays_next_save(self):
        vf = VarietyFilter(background_interval_sec=60.0)
        vf.should_save_background()  # first save
        vf._last_background_save_time -= 61.0  # simulate interval passed
        vf.reset_background_timer()  # reset — should delay again
        assert vf.should_save_background() is False
