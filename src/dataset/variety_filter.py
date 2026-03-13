"""
Variety filter: decides whether a new frame is different enough
from the last saved frame to be worth keeping.

Strategy:
  1. Resize frame to a small thumbnail for fast comparison.
  2. Compute a normalized HSV histogram.
  3. Compare with the last-saved histogram via Bhattacharyya distance.
  4. Also enforce a minimum time gap between saves.
  5. Save if distance > threshold OR enough time has passed (keep slow-moving cats).
"""

import time
from typing import Optional
import cv2
import numpy as np


class VarietyFilter:
    """
    Filters out near-duplicate frames to ensure dataset variety.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.15,
        min_interval_sec: float = 2.0,
        max_interval_sec: float = 30.0,
        thumb_size: tuple = (64, 64),
    ):
        self.similarity_threshold = similarity_threshold
        self.min_interval_sec = min_interval_sec
        self.max_interval_sec = max_interval_sec
        self.thumb_size = thumb_size

        self._last_histogram: Optional[np.ndarray] = None
        self._last_save_time: float = 0.0

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        thumb = cv2.resize(frame, self.thumb_size)
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        h_bins, s_bins = 16, 8
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [h_bins, s_bins],
            [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def should_save(self, frame: np.ndarray) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_save_time

        # Respect minimum interval
        if elapsed < self.min_interval_sec:
            return False

        hist = self._compute_histogram(frame)

        if self._last_histogram is None:
            # First frame ever — always save
            self._accept(hist, now)
            return True

        distance = cv2.compareHist(
            self._last_histogram, hist, cv2.HISTCMP_BHATTACHARYYA
        )

        # Force-save if max interval exceeded (even if cat hasn't moved)
        if elapsed >= self.max_interval_sec:
            self._accept(hist, now)
            return True

        # Save if frame is different enough
        if distance >= self.similarity_threshold:
            self._accept(hist, now)
            return True

        return False

    def _accept(self, hist: np.ndarray, timestamp: float) -> None:
        self._last_histogram = hist
        self._last_save_time = timestamp

    def reset(self) -> None:
        self._last_histogram = None
        self._last_save_time = 0.0
