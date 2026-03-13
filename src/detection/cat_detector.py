from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# COCO class ID for 'cat'
CAT_CLASS_ID = 15

# COCO class names for debug printing
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}


@dataclass
class Detection:
    """A single cat detection result."""
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    class_id: int = CAT_CLASS_ID

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class DetectionResult:
    """Result from processing one frame."""
    detections: List[Detection] = field(default_factory=list)
    frame: Optional[np.ndarray] = None

    @property
    def has_cats(self) -> bool:
        return len(self.detections) > 0

    @property
    def cat_count(self) -> int:
        return len(self.detections)


class CatDetector:
    """
    Wraps a YOLO model to detect cats in frames.
    Uses YOLOv8n by default — fast enough for real-time on CPU.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.imgsz = imgsz
        self._model = None
        self._model_path = model_path

    def _load_model(self):
        """Lazy-load model on first use."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a single frame and return cat detections."""
        self._load_model()

        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=[CAT_CLASS_ID],
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                        )
                    )

        return DetectionResult(detections=detections, frame=frame)

    def detect_all(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Run inference without any class filter and return all detections.
        Returns a dict of {class_name: best_confidence} for debug purposes.
        """
        self._load_model()

        results = self._model.predict(
            frame,
            conf=0.1,  # low threshold to catch weak detections
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        found: Dict[str, float] = {}
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = COCO_NAMES.get(class_id, f"class_{class_id}")
                    # Keep highest confidence per class
                    if name not in found or conf > found[name]:
                        found[name] = conf
        return found

    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes on a copy of the frame."""
        import cv2
        out = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"cat {det.confidence:.2f}"
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return out
