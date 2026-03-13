"""
DatasetPipeline: orchestrates preprocessing, detection, variety filtering, and saving.
Designed to be called frame-by-frame from the main loop.
"""

from typing import Optional, Tuple
import numpy as np

from src.detection.cat_detector import CatDetector, DetectionResult
from src.detection.preprocessor import CLAHEPreprocessor
from src.dataset.variety_filter import VarietyFilter
from src.dataset.saver import DatasetSaver


class DatasetPipeline:
    """
    High-level pipeline that:
      1. Applies CLAHE preprocessing to correct overexposure.
      2. Runs YOLO detection on the preprocessed frame.
      3. Checks variety filter (skip near-duplicates).
      4. Saves qualifying preprocessed frames + metadata to disk.

    Saving the preprocessed frame (not the raw one) ensures the fine-tuning
    dataset matches the input distribution seen at inference time.
    """

    def __init__(
        self,
        detector: CatDetector,
        variety_filter: VarietyFilter,
        saver: DatasetSaver,
        preprocessor: Optional[CLAHEPreprocessor] = None,
    ):
        self.detector = detector
        self.variety_filter = variety_filter
        self.saver = saver
        self.preprocessor = preprocessor or CLAHEPreprocessor(enabled=False)

        self._total_frames: int = 0
        self._detection_frames: int = 0
        self._saved_frames: int = 0

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[DetectionResult]]:

        self._total_frames += 1

        preprocessed = self.preprocessor.process(frame)
        result = self.detector.detect(preprocessed)

        if not result.has_cats:
            return preprocessed, None

        self._detection_frames += 1

        if not self.variety_filter.should_save(preprocessed):
            return preprocessed, None

        self.saver.save(result, preprocessed)
        self._saved_frames += 1
        return preprocessed, result

    @property
    def stats(self) -> dict:
        return {
            "total_frames": self._total_frames,
            "detection_frames": self._detection_frames,
            "saved_frames": self._saved_frames,
        }
