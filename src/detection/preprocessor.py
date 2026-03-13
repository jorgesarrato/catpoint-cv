"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessor.

Applies CLAHE to the luminance channel (L in LAB colorspace) to correct
overexposure and improve local contrast, while preserving color information.
  - Applying CLAHE to the V channel of HSV tends to produce color artifacts
    in overexposed regions.

For IR / night mode frames (near-grayscale):
  - CLAHE still helps by recovering detail in blown-out bright regions.
  - The effect is applied consistently regardless of whether the frame is
    color or IR, so training and inference inputs are always treated the same way.
"""

import cv2
import numpy as np


class CLAHEPreprocessor:
    """
    Applies CLAHE to the L channel of an LAB frame.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
        enabled: bool = True,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.enabled = enabled
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to a BGR frame and return the enhanced BGR frame.
        The input frame is not modified.
        """
        if not self.enabled:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
