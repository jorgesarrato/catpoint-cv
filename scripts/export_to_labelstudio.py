"""
scripts/export_to_labelstudio.py

Converts dataset metadata JSONs into a Label Studio import file.

- Cat detection images: pre-loaded with bounding boxes (label='cat')
- Background images: imported as blank tasks for manual annotation
- Skips images already present in a Label Studio export file

Usage:
    python scripts/export_to_labelstudio.py
    python scripts/export_to_labelstudio.py \\
        --skip-exported data/labelstudio_merged.json
    python scripts/export_to_labelstudio.py \\
        --data data/raw \\
        --output data/labelstudio_import.json \\
        --skip-exported data/labelstudio_merged.json \\
        --document-root /home/jsarrato/PersonalProjects/catpoint-cv

Label Studio setup:
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/path/to/your/project
    label-studio start
"""

import argparse
import json
from pathlib import Path

import cv2


# Maps class_id from fine-tuned model to Label Studio label name.
# class_id 15 is the COCO 'cat' class from the pretrained model.
CLASS_ID_TO_LABEL: dict[int, str] = {
    0: "salo",
    1: "taro",
    15: "cat",
}


def load_labeled_filenames(export_path: Path) -> set:
    """Extract image filenames already present in a Label Studio export JSON."""
    if not export_path.exists():
        return set()
    data = json.loads(export_path.read_text())
    if isinstance(data, dict):
        for key in ("tasks", "annotations", "results"):
            if key in data:
                data = data[key]
                break
    filenames = set()
    for task in data:
        if not isinstance(task, dict):
            continue
        url = task.get("data", {}).get("image", "")
        if url:
            filename = Path(url.split("?d=")[-1]).name if "?d=" in url else Path(url).name
            filenames.add(filename)
    return filenames


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


def make_task(image_path: Path, detections: list, base_url: str = "", document_root: str = ".") -> dict:
    abs_path = image_path.resolve()
    if base_url:
        rel = abs_path.relative_to(Path(document_root).resolve())
        image_url = f"{base_url.rstrip('/')}/{rel}"
    else:
        rel = abs_path.relative_to(Path(document_root).resolve())
        image_url = f"/data/local-files/?d={rel}"
    w, h = image_dimensions(image_path)

    results = []
    for det in detections:
        coords = bbox_to_percent(det["bbox"], w, h)
        class_id = det.get("class_id")
        label = CLASS_ID_TO_LABEL.get(class_id, "cat") if class_id is not None else "cat"
        results.append({
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": w,
            "original_height": h,
            "value": {
                **coords,
                "rectanglelabels": [label],
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
    p.add_argument("--skip-exported", default="",
                   help="Label Studio export JSON to use as skip list "
                        "(e.g. data/labelstudio_merged.json)")
    p.add_argument("--base-url", default="",
                   help="Base URL for images, e.g. http://localhost:8081 "
                        "(uses /data/local-files/ if not set)")
    p.add_argument("--document-root", default=".",
                   help="Must match LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (default: .)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)
    output_path = Path(args.output)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load already-labeled filenames from export if provided
    skip_filenames: set = set()
    if args.skip_exported:
        skip_filenames = load_labeled_filenames(Path(args.skip_exported))
        print(f"Skipping {len(skip_filenames)} already-labeled images from {args.skip_exported}")

    meta_files = collect_meta_files(data_dir)
    tasks = []
    skipped = 0
    errors = 0

    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text())
        image_path = Path(meta["full_frame"])

        if not image_path.exists():
            print(f"  [WARN] Image not found, skipping: {image_path}")
            errors += 1
            continue

        if image_path.name in skip_filenames:
            skipped += 1
            continue

        try:
            task = make_task(image_path, meta.get("detections", []),
                             base_url=args.base_url, document_root=args.document_root)
            tasks.append(task)
        except Exception as e:
            print(f"  [WARN] Failed to process {image_path.name}: {e}")
            errors += 1

    if not tasks:
        print("No new images to export.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2))

    print(f"Exported  : {len(tasks)} tasks -> {output_path}")
    print(f"Skipped   : {skipped} already labeled")
    print(f"Errors    : {errors}")
    print(f"\nImport into Label Studio: Projects -> Import -> upload {output_path}")


if __name__ == "__main__":
    main()
