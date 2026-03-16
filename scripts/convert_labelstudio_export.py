"""
scripts/convert_labelstudio_export.py

Converts a Label Studio export JSON into YOLO format for fine-tuning.

Label Studio export format (per task):
  {
    "data": {"image": "/data/local-files/?d=data/raw/img.jpg"},
    "annotations": [{
      "result": [{
        "value": {
          "x": 10.5, "y": 20.3, "width": 15.2, "height": 25.1,
          "rectanglelabels": ["salo"]
        },
        "original_width": 1920,
        "original_height": 1080
      }]
    }]
  }

YOLO output format (per line in .txt):
  class_id cx cy width height   (all normalised 0-1)

Background images (no annotations) get an empty .txt file —
YOLO treats these as hard negative samples during training.

Usage:
    python scripts/convert_labelstudio_export.py --export data/labelstudio_export.json
    python scripts/convert_labelstudio_export.py \\
        --export data/labelstudio_export.json \\
        --output data/labeled \\
        --classes salo taro
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, parse_qs


CLASS_NAMES = ["salo", "taro"]


def parse_image_path(image_url: str, document_root: str = ".") -> Path:
    """
    Extract the filesystem path from a Label Studio image URL.

    Handles both formats:
      /data/local-files/?d=data/raw/img.jpg  -> <document_root>/data/raw/img.jpg
      http://localhost:8081/data/raw/img.jpg -> <document_root>/data/raw/img.jpg
    """
    if "/data/local-files/" in image_url:
        parsed = urlparse(image_url)
        qs = parse_qs(parsed.query)
        rel = qs.get("d", [""])[0]
        return Path(document_root) / rel
    else:
        # http server URL — strip the base URL and reconstruct path
        parsed = urlparse(image_url)
        rel = parsed.path.lstrip("/")
        return Path(document_root) / rel


def percent_to_yolo(x_pct: float, y_pct: float,
                    w_pct: float, h_pct: float) -> tuple[float, float, float, float]:
    """
    Convert Label Studio percentage coords to YOLO normalised cx/cy/w/h.

    Label Studio: x, y are top-left corner as percentage of image dimensions.
    YOLO: cx, cy are centre as fraction of image dimensions.
    """
    cx = (x_pct + w_pct / 2) / 100
    cy = (y_pct + h_pct / 2) / 100
    w  = w_pct / 100
    h  = h_pct / 100
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def convert_task(
    task: dict,
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    document_root: str,
) -> tuple[int, list[str]]:
    """
    Convert one Label Studio task to a YOLO image + label file.

    Returns (n_boxes_written, warnings).
    """
    warnings = []
    image_url = task.get("data", {}).get("image", "")
    if not image_url:
        return 0, ["Task has no image URL, skipping"]

    src_path = parse_image_path(image_url, document_root)
    if not src_path.exists():
        return 0, [f"Image not found: {src_path}"]

    stem = src_path.stem
    dest_image = images_dir / src_path.name
    dest_label = labels_dir / f"{stem}.txt"

    # Copy image
    shutil.copy2(src_path, dest_image)

    # Extract annotations — use first annotation set if multiple exist
    annotations = task.get("annotations", [])
    if not annotations or not annotations[0].get("result"):
        # Background image — empty label file (negative sample for YOLO)
        dest_label.write_text("")
        return 0, warnings

    results = annotations[0]["result"]
    lines = []

    for r in results:
        if r.get("type") != "rectanglelabels":
            continue
        value = r.get("value", {})
        labels = value.get("rectanglelabels", [])
        if not labels:
            warnings.append(f"{stem}: box with no label, skipping")
            continue

        label = labels[0]
        if label not in class_names:
            warnings.append(f"{stem}: unknown label '{label}', skipping")
            continue

        class_id = class_names.index(label)
        cx, cy, w, h = percent_to_yolo(
            value["x"], value["y"], value["width"], value["height"]
        )
        lines.append(f"{class_id} {cx} {cy} {w} {h}")

    dest_label.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines), warnings


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Label Studio export JSON to YOLO format"
    )
    p.add_argument("--export", default="data/labelstudio_export.json",
                   help="Label Studio export JSON file (default: data/labelstudio_export.json)")
    p.add_argument("--output", default="data/labeled",
                   help="Output directory for images/ and labels/ (default: data/labeled)")
    p.add_argument("--classes", nargs="+", default=CLASS_NAMES,
                   help=f"Class names in order (default: {CLASS_NAMES})")
    p.add_argument("--document-root", default=".",
                   help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (default: .)")
    return p.parse_args()


def main():
    args = parse_args()

    export_path = Path(args.export)
    if not export_path.exists():
        raise FileNotFoundError(f"Export file not found: {export_path}")

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(export_path.read_text())
    print(tasks)

    # Handle both list-of-dicts and wrapped formats
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", tasks.get("annotations", [tasks]))

    print(f"Processing {len(tasks)} tasks...")
    if tasks:
        print(f"  First task type: {type(tasks[0])}")
        print(tasks)
        if isinstance(tasks[0], dict):
            print(f"  First task keys: {list(tasks[0].keys())}")

    class_counts: Counter = Counter()
    total_boxes = 0
    background_count = 0
    all_warnings = []

    for task in tasks:
        n_boxes, warnings = convert_task(
            task, images_dir, labels_dir, args.classes, args.document_root
        )
        all_warnings.extend(warnings)

        if n_boxes == 0:
            background_count += 1
        else:
            total_boxes += n_boxes
            # Count per class
            annotations = task.get("annotations", [])
            if annotations:
                for r in annotations[0].get("result", []):
                    labels = r.get("value", {}).get("rectanglelabels", [])
                    if labels and labels[0] in args.classes:
                        class_counts[labels[0]] += 1

    print(f"\n--- Conversion Summary ---")
    print(f"  Tasks processed  : {len(tasks)}")
    print(f"  Boxes written    : {total_boxes}")
    print(f"  Background images: {background_count}")
    for cls in args.classes:
        print(f"  {cls:12s}     : {class_counts[cls]} boxes")
    if all_warnings:
        print(f"\n  Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"    - {w}")
    print(f"\n  Output: {output_dir}")
    print(f"\nNext step: python scripts/split_dataset.py --input {output_dir}")


if __name__ == "__main__":
    main()
