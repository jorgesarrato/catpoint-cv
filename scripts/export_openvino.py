"""
scripts/export_openvino.py

Exports a fine-tuned YOLOv8 .pt checkpoint to OpenVINO FP32 or INT8 format.

INT8 quantization requires a calibration dataset (~100-200 representative images).
Use your labeled dataset directory for calibration — the more images the better.

Usage:
    # FP32 (default)
    python scripts/export_openvino.py --model best.pt

    # INT8 (requires calibration data)
    python scripts/export_openvino.py --model best.pt --int8 --data data/labeled/dataset.yaml

    python scripts/export_openvino.py --model best.pt --imgsz 1280 --output models/
"""

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # allows importing the module in tests without ultralytics


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 checkpoint to OpenVINO")
    p.add_argument("--model", required=True,
                   help="Path to fine-tuned .pt checkpoint")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Inference image size — must match training imgsz (default: 1280)")
    p.add_argument("--output", default="models",
                   help="Directory to copy the exported model into (default: models/)")
    p.add_argument("--int8", action="store_true",
                   help="Export INT8 quantized model (faster, requires --data for calibration)")
    p.add_argument("--data", default="",
                   help="Path to dataset.yaml for INT8 calibration (required with --int8)")
    return p.parse_args()


def main():
    args = parse_args()

    if YOLO is None:
        raise ImportError("ultralytics is required: pip install ultralytics")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if model_path.suffix != ".pt":
        raise ValueError(f"Expected a .pt file, got: {model_path}")

    if args.int8 and not args.data:
        raise ValueError(
            "--data is required for INT8 export. "
            "Pass your dataset.yaml, e.g. --data data/labeled/dataset.yaml"
        )

    precision = "INT8" if args.int8 else "FP32"
    print(f"Loading checkpoint: {model_path}")
    model = YOLO(str(model_path))

    print(f"Exporting to OpenVINO {precision} (imgsz={args.imgsz})...")

    export_kwargs = dict(
        format="openvino",
        imgsz=args.imgsz,
        half=False,
        task="detect",
    )
    if args.int8:
        export_kwargs["int8"] = True
        export_kwargs["data"] = args.data

    model.export(**export_kwargs)

    # Ultralytics names the export directory differently for INT8 vs FP32
    possible_dirs = [
        model_path.parent / (model_path.stem + "_int8_openvino_model"),
        model_path.parent / (model_path.stem + "_openvino_model"),
    ]
    exported_dir = next((d for d in possible_dirs if d.exists()), None)
    if exported_dir is None:
        raise RuntimeError(
            f"Export succeeded but no output directory found. Looked for: "
            f"{[str(d) for d in possible_dirs]}"
        )

    # Copy to output directory with precision suffix to avoid overwriting FP32
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_int8" if args.int8 else "_fp32"
    dest_name = model_path.stem + suffix + "_openvino_model"
    dest = output_dir / dest_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(exported_dir, dest)

    print(f"\nExport complete ({precision}).")
    print(f"Model saved to: {dest}")
    print(f"\nRun inference with:")
    print(f"  python main.py --model {dest} --imgsz {args.imgsz}")


if __name__ == "__main__":
    main()
