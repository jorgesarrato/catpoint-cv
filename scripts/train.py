"""
scripts/train.py

Fine-tunes a pretrained YOLOv8 model on the cat identity dataset.

Design decisions:
  - Starts from pretrained yolov8m.pt (keeps COCO knowledge)
  - Freezes backbone (first 10 layers) to prevent overfitting on small datasets
  - imgsz=1280 to match inference setup and improve small-object detection
  - Runs on CPU locally or GPU on Colab (auto-detected via --device)

Usage (local CPU):
    python scripts/train.py --data data/labeled/dataset.yaml

Usage (Colab GPU):
    python scripts/train.py --data /content/labeled/dataset.yaml --device 0

Output:
    runs/train/<name>/weights/best.pt   <- use this for inference
    runs/train/<name>/weights/last.pt
"""

import argparse
from pathlib import Path


BACKBONE_FREEZE_LAYERS = 10  # freeze first 10 layers of YOLOv8 backbone


def freeze_backbone(model, n_layers: int) -> None:
    """Freeze the first n_layers of the model to prevent overfitting."""
    freeze = [f"model.{i}." for i in range(n_layers)]
    for name, param in model.model.named_parameters():
        if any(name.startswith(f) for f in freeze):
            param.requires_grad = False
    frozen = sum(
        1 for name, p in model.model.named_parameters()
        if not p.requires_grad
    )
    total = sum(1 for _ in model.model.named_parameters())
    print(f"Frozen {frozen}/{total} parameter tensors (first {n_layers} layers)")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on cat identity dataset")
    p.add_argument("--data", default="data/labeled/dataset.yaml",
                   help="Path to dataset.yaml (default: data/labeled/dataset.yaml)")
    p.add_argument("--model", default="yolov8m.pt",
                   help="Pretrained model to fine-tune (default: yolov8m.pt)")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs (default: 100)")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (default: 8, reduce if OOM)")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Training image size (default: 1280)")
    p.add_argument("--device", default="cpu",
                   help="Device: cpu or 0 for GPU (default: cpu)")
    p.add_argument("--name", default="cat_identity",
                   help="Run name for output directory (default: cat_identity)")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience in epochs (default: 20)")
    p.add_argument("--no-freeze", action="store_true",
                   help="Disable backbone freezing (not recommended for small datasets)")
    return p.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {data_path}. "
            "Run scripts/split_dataset.py first."
        )

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    if not args.no_freeze:
        freeze_backbone(model, BACKBONE_FREEZE_LAYERS)

    print(f"\nStarting fine-tuning:")
    print(f"  data    : {args.data}")
    print(f"  epochs  : {args.epochs}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  batch   : {args.batch}")
    print(f"  device  : {args.device}")
    print(f"  patience: {args.patience}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        # Augmentation — conservative for small datasets
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,   # disable mixup — unreliable with only 2 classes
        # Regularisation
        dropout=0.1,
        weight_decay=0.0005,
        # Logging
        plots=True,
        save=True,
        save_period=10,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best checkpoint: {best}")
    print(f"\nNext step: download {best} and run:")
    print(f"  python scripts/export_openvino.py --model {best}")


if __name__ == "__main__":
    main()
