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


def _read_num_classes(model_path: "Path", fallback: int = 80) -> int:
    """
    Read the number of classes from a model directory's metadata yaml.
    Ultralytics writes a metadata.yaml (or similar) into every OpenVINO export.
    Falls back to reading from args.yaml in the training run directory.
    Returns `fallback` if nothing can be found.
    """
    import yaml
    from pathlib import Path

    path = Path(model_path)
    # For OpenVINO directories, check for metadata yaml files
    candidates = []
    if path.is_dir():
        candidates += list(path.glob("*.yaml"))
    # Also check parent for training args.yaml
    candidates += list(path.parent.glob("*.yaml"))

    for yaml_path in candidates:
        try:
            data = yaml.safe_load(yaml_path.read_text())
            if isinstance(data, dict) and "nc" in data:
                return int(data["nc"])
        except Exception:
            continue
    return fallback


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
        self._is_openvino: bool = False
        self._num_classes: int = 80

    def _load_model(self):
        """
        Lazy-load model on first use.

        If model_path points to a .pt file, exports it to OpenVINO FP32 format
        on first run (produces a <stem>_openvino_model/ directory alongside the
        .pt file), then loads the exported model. Subsequent runs skip the export
        and load directly from the OpenVINO directory.

        If model_path already points to an OpenVINO directory, loads it directly.
        Sets self._is_openvino so other methods can adapt their behaviour.
        """
        if self._model is not None:
            return

        from ultralytics import YOLO
        from pathlib import Path

        path = Path(self._model_path)

        if path.suffix == ".pt":
            openvino_dir = path.parent / (path.stem + "_openvino_model")
            if not openvino_dir.exists():
                pt_model = YOLO(self._model_path)
                pt_model.export(
                    format="openvino",
                    half=False,
                    imgsz=self.imgsz,
                    task="detect",
                )
            self._model = YOLO(str(openvino_dir), task="detect")
            self._is_openvino = True
        else:
            self._model = YOLO(self._model_path, task="detect")
            self._is_openvino = True

        # Detect number of classes — fine-tuned models have fewer than 80
        # and must not be filtered by COCO class IDs.
        # Try yaml first, then model info dict, then model attribute.
        nc = _read_num_classes(Path(self._model_path), fallback=None)
        if nc is None:
            try:
                nc = len(self._model.names)
            except Exception:
                nc = 80
        self._num_classes = nc
        print(f"[CatDetector] Loaded model with {self._num_classes} class(es): "
              f"{getattr(self._model, 'names', {})}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a single frame and return cat detections."""
        self._load_model()

        # OpenVINO's native runtime requires a C-contiguous array.
        # Frames from the stream thread may be non-contiguous slices.
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=[CAT_CLASS_ID] if self._num_classes == 80 else None,
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
                    class_id = int(box.cls[0])  # Extract the class ID

                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=class_id   # Pass it to the dataclass
                        )
                    )

        return DetectionResult(detections=detections, frame=frame)

    def detect_all(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Run inference and return all detections for debug purposes.
        Returns a dict of {class_name: best_confidence}.
        """
        self._load_model()

        is_openvino = self._is_openvino

        predict_kwargs = dict(
            conf=0.1,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not is_openvino and self._num_classes == 80:
            predict_kwargs["classes"] = None  # unfiltered — all 80 COCO classes

        results = self._model.predict(frame, **predict_kwargs)

        found: Dict[str, float] = {}
        
        # Grab the custom names dict from the model, fallback to COCO_NAMES just in case
        model_names = getattr(self._model, 'names', COCO_NAMES)

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Use the model's names instead of hardcoded COCO_NAMES
                    name = model_names.get(class_id, f"class_{class_id}")
                    
                    if name not in found or conf > found[name]:
                        found[name] = conf
        return found
    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes on a copy of the frame."""
        import cv2
        out = frame.copy()
        
        model_names = getattr(self._model, 'names', COCO_NAMES)
        
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            class_name = model_names.get(det.class_id, f"class_{det.class_id}")
            
            label = f"{class_name} {det.confidence:.2f}"
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return out
