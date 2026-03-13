"""
Saves cat detection frames to disk with rich metadata.

Directory layout:
  data/raw/
    YYYYMMDD_HHMMSS_fff_<n>cats.jpg      full frame
    YYYYMMDD_HHMMSS_fff_crop_0.jpg       first cat crop
    YYYYMMDD_HHMMSS_fff_crop_1.jpg       second cat crop (if present)
    YYYYMMDD_HHMMSS_fff_meta.json        detection metadata
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.detection.cat_detector import DetectionResult


class DatasetSaver:
    """Persists frames and crops for detected cat events."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        save_crops: bool = True,
        jpg_quality: int = 95,
        min_crop_px: int = 32,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_crops = save_crops
        self.jpg_quality = jpg_quality
        self.min_crop_px = min_crop_px
        self._session_count: int = 0

    def save(self, result: DetectionResult, frame: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Save a detection event.
        """
        img = frame if frame is not None else result.frame
        if img is None or not result.has_cats:
            return None

        ts = datetime.now()
        stem = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
        stem = f"{stem}_{result.cat_count}cats"

        # Save full frame
        full_path = self.output_dir / f"{stem}.jpg"
        cv2.imwrite(
            str(full_path), img,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
        )

        # Save individual crops
        crops_saved = []
        if self.save_crops:
            h, w = img.shape[:2]
            for i, det in enumerate(result.detections):
                x1, y1, x2, y2 = det.bbox
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) < self.min_crop_px or (y2 - y1) < self.min_crop_px:
                    continue
                crop = img[y1:y2, x1:x2]
                crop_path = self.output_dir / f"{stem}_crop_{i}.jpg"
                cv2.imwrite(
                    str(crop_path), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                )
                crops_saved.append(str(crop_path))

        # Save metadata
        meta = {
            "timestamp": ts.isoformat(),
            "cat_count": result.cat_count,
            "full_frame": str(full_path),
            "crops": crops_saved,
            "detections": [
                {
                    "bbox": list(det.bbox),
                    "confidence": round(det.confidence, 4),
                }
                for det in result.detections
            ],
        }
        meta_path = self.output_dir / f"{stem}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        self._session_count += 1
        return stem

    @property
    def session_count(self) -> int:
        return self._session_count
