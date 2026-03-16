"""
scripts/export_to_labelstudio.py

Converts dataset metadata JSONs into a Label Studio import file.

- Cat detection images: pre-loaded with bounding boxes (label='cat')
- Background images: imported as blank tasks for manual annotation
- Skips images already tracked in data/labelstudio_processed.json

Usage:
    python scripts/export_to_labelstudio.py
    python scripts/export_to_labelstudio.py --data data/raw --output data/labelstudio_import.json

Label Studio setup:
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/path/to/your/project
    label-studio start
"""

import argparse
import json
from pathlib import Path

import cv2


TRACKING_FILE = "data/labelstudio_processed.json"


def load_processed(tracking_path: Path) -> set:
    if tracking_path.exists():
        return set(json.loads(tracking_path.read_text()))
    return set()


def save_processed(tracking_path: Path, processed: set) -> None:
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_path.write_text(json.dumps(sorted(processed), indent=2))


def image_dimensions(image_path: Path) -> tuple[int, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def bbox_to_percent(bbox: list, w: int, h: int) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "x": x1 / w * 100,
        "y": y1 / h * 100,
        "width": (x2 - x1) / w * 100,
        "height": (y2 - y1) / h * 100,
    }


def make_task(image_path: Path, detections: list) -> dict:
    abs_path = image_path.resolve()
    image_url = f"/data/local-files/?d={abs_path}"
    w, h = image_dimensions(image_path)

    results = []
    for det in detections:
        coords = bbox_to_percent(det["bbox"], w, h)
        results.append({
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": w,
            "original_height": h,
            "value": {
                **coords,
                "rectanglelabels": ["cat"],
            },
        })

    task = {"data": {"image": image_url}}
    if results:
        task["annotations"] = [{"result": results}]
    return task


def collect_meta_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*_meta.json"))


def parse_args():
    p = argparse.ArgumentParser(description="Export dataset to Label Studio import format")
    p.add_argument("--data", default="data/raw",
                   help="Directory containing images and metadata JSONs (default: data/raw)")
    p.add_argument("--output", default="data/labelstudio_import.json",
                   help="Output Label Studio import file (default: data/labelstudio_import.json)")
    p.add_argument("--tracking", default=TRACKING_FILE,
                   help=f"Tracking file for processed images (default: {TRACKING_FILE})")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)
    output_path = Path(args.output)
    tracking_path = Path(args.tracking)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    processed = load_processed(tracking_path)
    meta_files = collect_meta_files(data_dir)

    tasks = []
    skipped = 0
    errors = 0
    newly_processed = set()

    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text())
        image_path = Path(meta["full_frame"])

        if not image_path.exists():
            print(f"  [WARN] Image not found, skipping: {image_path}")
            errors += 1
            continue

        image_name = image_path.name
        if image_name in processed:
            skipped += 1
            continue

        try:
            task = make_task(image_path, meta.get("detections", []))
            tasks.append(task)
            newly_processed.add(image_name)
        except Exception as e:
            print(f"  [WARN] Failed to process {image_name}: {e}")
            errors += 1

    if not tasks:
        print("No new images to export.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2))

    processed.update(newly_processed)
    save_processed(tracking_path, processed)

    print(f"Exported  : {len(tasks)} tasks -> {output_path}")
    print(f"Skipped   : {skipped} already processed")
    print(f"Errors    : {errors}")
    print(f"Tracking  : {tracking_path} ({len(processed)} total processed)")
    print(
        "\nImport into Label Studio: Projects -> Import -> upload "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
