"""
scripts/merge_labelstudio_exports.py

Merges multiple Label Studio export JSONs into a single file,
deduplicating by image filename.

Export files are typically named like:
    project-3-at-2026-03-15-21-02-4afb93f8.json

Usage:
    python scripts/merge_labelstudio_exports.py
    python scripts/merge_labelstudio_exports.py --input data/exports/ --output data/labelstudio_merged.json
    python scripts/merge_labelstudio_exports.py --input data/exports/project-*.json
"""

import argparse
import json
from pathlib import Path


def extract_image_filename(task: dict) -> str | None:
    """Extract the image filename from a Label Studio task."""
    url = task.get("data", {}).get("image", "")
    if not url:
        return None
    return Path(url.split("?d=")[-1]).name if "?d=" in url else Path(url).name


def load_export(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        # Some exports wrap tasks in a dict
        if isinstance(data, dict):
            for key in ("tasks", "annotations", "results"):
                if key in data:
                    return data[key]
        return []
    except Exception as e:
        print(f"  [WARN] Failed to load {path.name}: {e}")
        return []


def merge_exports(export_files: list[Path]) -> tuple[list[dict], dict]:
    """
    Merge export files, deduplicating by image filename.
    Returns (merged_tasks, stats).
    """
    seen: dict[str, dict] = {}  # filename -> task
    stats = {"files": 0, "total": 0, "duplicates": 0}

    for path in sorted(export_files):
        tasks = load_export(path)
        stats["files"] += 1
        stats["total"] += len(tasks)
        print(f"  {path.name}: {len(tasks)} tasks")

        for task in tasks:
            filename = extract_image_filename(task)
            if filename is None:
                continue
            if filename in seen:
                stats["duplicates"] += 1
            else:
                seen[filename] = task

    return list(seen.values()), stats


def collect_export_files(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.glob("*.json"))
    # Treat as glob pattern
    parent = p.parent
    pattern = p.name
    return sorted(parent.glob(pattern))


def parse_args():
    p = argparse.ArgumentParser(description="Merge Label Studio export JSONs")
    p.add_argument("--input", default="data/exports",
                   help="Directory of export JSONs or glob pattern (default: data/exports)")
    p.add_argument("--output", default="data/labelstudio_merged.json",
                   help="Output merged JSON (default: data/labelstudio_merged.json)")
    return p.parse_args()


def main():
    args = parse_args()

    export_files = collect_export_files(args.input)
    if not export_files:
        raise FileNotFoundError(f"No export JSON files found at: {args.input}")

    print(f"Found {len(export_files)} export file(s):")
    merged, stats = merge_exports(export_files)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))

    print(f"\n--- Merge Summary ---")
    print(f"  Files processed : {stats['files']}")
    print(f"  Total tasks     : {stats['total']}")
    print(f"  Duplicates      : {stats['duplicates']}")
    print(f"  Unique tasks    : {len(merged)}")
    print(f"  Output          : {output_path}")


if __name__ == "__main__":
    main()
