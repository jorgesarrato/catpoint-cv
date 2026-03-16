"""
scripts/split_dataset.py

Splits a flat directory of labeled images + YOLO txt files into
train/val/test subsets and writes dataset.yaml.

Expected input structure:
    data/labeled/
        images/
            img1.jpg
            img2.jpg
        labels/
            img1.txt
            img2.txt

Output structure:
    data/labeled/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/
        dataset.yaml

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --input data/labeled --split 0.8 0.1 0.1
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path


CLASS_NAMES = ["salo", "taro"]


def collect_samples(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs that have both files present."""
    samples = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            samples.append((img_path, label_path))
        else:
            print(f"  [WARN] No label for {img_path.name}, skipping")
    return samples


def split_samples(
    samples: list,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def copy_split(samples: list, dest_dir: Path) -> None:
    (dest_dir / "images").mkdir(parents=True, exist_ok=True)
    (dest_dir / "labels").mkdir(parents=True, exist_ok=True)
    for img_path, label_path in samples:
        shutil.copy2(img_path, dest_dir / "images" / img_path.name)
        shutil.copy2(label_path, dest_dir / "labels" / label_path.name)


def write_dataset_yaml(output_dir: Path, class_names: list[str]) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    content = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path.write_text(yaml.dump(content, default_flow_style=False))
    return yaml_path


def parse_args():
    p = argparse.ArgumentParser(description="Split labeled dataset into train/val/test")
    p.add_argument("--input", default="data/labeled",
                   help="Input directory containing images/ and labels/ (default: data/labeled)")
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Split ratios (default: 0.8 0.1 0.1)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--classes", nargs="+", default=CLASS_NAMES,
                   help=f"Class names (default: {CLASS_NAMES})")
    return p.parse_args()


def main():
    args = parse_args()

    train_r, val_r, test_r = args.split
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    input_dir = Path(args.input)
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected {images_dir} and {labels_dir} to exist. "
            "Run convert_labelstudio_export.py first."
        )

    print(f"Collecting samples from {input_dir}...")
    samples = collect_samples(images_dir, labels_dir)
    if not samples:
        raise ValueError("No labeled samples found.")
    print(f"Found {len(samples)} labeled samples")

    train, val, test = split_samples(samples, train_r, val_r, seed=args.seed)

    print(f"\nSplit: {len(train)} train / {len(val)} val / {len(test)} test")

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        dest = input_dir / split_name
        copy_split(split_data, dest)
        print(f"  Copied {len(split_data)} samples to {dest}")

    yaml_path = write_dataset_yaml(input_dir, args.classes)
    print(f"\nDataset YAML written to {yaml_path}")
    print(f"Classes: {args.classes}")


if __name__ == "__main__":
    main()
