"""
scripts/benchmark.py

Benchmarks YOLO inference latency across model sizes, image sizes, and precision.

Runs each configuration for a fixed number of iterations (with warmup) and reports
per-frame latency stats (mean, p50, p95, p99) plus throughput in FPS.

Usage:
    # Quick benchmark with a single frame from disk
    python scripts/benchmark.py --image data/raw/20260313_161054_885_1cats.jpg

    # Benchmark specific models
    python scripts/benchmark.py --image frame.jpg --models yolov8s.pt yolov8m.pt

    # Benchmark OpenVINO exported models
    python scripts/benchmark.py --image frame.jpg --models models/best_fp32_openvino_model models/best_int8_openvino_model

    # Customize iterations and image sizes
    python scripts/benchmark.py --image frame.jpg --imgsz 640 1280 --warmup 10 --iterations 50

    # Use a synthetic 1920x1080 frame (no --image needed)
    python scripts/benchmark.py

    # Export results to CSV
    python scripts/benchmark.py --csv results.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark YOLO inference latency")
    p.add_argument(
        "--models", nargs="+",
        default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="Model paths to benchmark (default: yolov8n/s/m)",
    )
    p.add_argument(
        "--imgsz", nargs="+", type=int, default=[640, 1280],
        help="Image sizes to test (default: 640 1280)",
    )
    p.add_argument(
        "--image", default=None,
        help="Path to a test image (default: synthetic 1920x1080 frame)",
    )
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    p.add_argument("--iterations", type=int, default=30, help="Timed iterations (default: 30)")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    p.add_argument("--csv", default=None, help="Export results to CSV file")
    return p.parse_args()


def load_frame(image_path):
    """Load a test frame, or generate a synthetic 1920x1080 one."""
    if image_path:
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: cannot read image: {image_path}")
            sys.exit(1)
        h, w = frame.shape[:2]
        print(f"Using image: {image_path} ({w}x{h})")
        return frame
    else:
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        print("Using synthetic frame: 1920x1080")
        return frame


def benchmark_model(model_path, frame, imgsz, conf, warmup, iterations):
    """Run benchmark for a single model+imgsz combo. Returns latency list in ms."""
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        return None, f"not found: {model_path}"

    model = YOLO(str(model_path), task="detect")

    # Determine precision from path name
    precision = "FP32"
    name_lower = str(model_path).lower()
    if "_int8" in name_lower:
        precision = "INT8"
    elif "_fp16" in name_lower or "half" in name_lower:
        precision = "FP16"
    elif name_lower.endswith(".pt"):
        precision = "PyTorch"

    # Warmup
    for _ in range(warmup):
        model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)

    # Timed runs
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    return latencies, precision


def print_results(results):
    """Pretty-print benchmark results table."""
    header = f"{'Model':<45} {'Prec':<8} {'imgsz':>5} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'FPS':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['model']:<45} {r['precision']:<8} {r['imgsz']:>5} "
            f"{r['mean_ms']:>7.1f}ms {r['p50_ms']:>7.1f}ms "
            f"{r['p95_ms']:>7.1f}ms {r['p99_ms']:>7.1f}ms "
            f"{r['fps']:>6.1f}"
        )
    print(sep)


def export_csv(results, csv_path):
    """Write results to CSV."""
    fields = ["model", "precision", "imgsz", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "fps"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nResults saved to: {csv_path}")


def main():
    args = parse_args()
    frame = load_frame(args.image)

    results = []
    total = len(args.models) * len(args.imgsz)
    current = 0

    for model_path in args.models:
        for imgsz in args.imgsz:
            current += 1
            model_name = str(model_path)
            # Truncate long paths for display
            if len(model_name) > 44:
                model_name = "..." + model_name[-41:]

            print(f"\n[{current}/{total}] {model_name} @ {imgsz}px", end="", flush=True)

            latencies, precision = benchmark_model(
                model_path, frame, imgsz, args.conf,
                args.warmup, args.iterations,
            )

            if latencies is None:
                print(f" — SKIPPED ({precision})")
                continue

            lat = np.array(latencies)
            result = {
                "model": model_name,
                "precision": precision,
                "imgsz": imgsz,
                "mean_ms": float(np.mean(lat)),
                "p50_ms": float(np.percentile(lat, 50)),
                "p95_ms": float(np.percentile(lat, 95)),
                "p99_ms": float(np.percentile(lat, 99)),
                "fps": 1000.0 / float(np.mean(lat)),
            }
            results.append(result)
            print(f" — {result['mean_ms']:.0f}ms ({result['fps']:.1f} FPS)")

    if results:
        print_results(results)
        if args.csv:
            export_csv(results, args.csv)
    else:
        print("\nNo models were benchmarked. Check model paths.")


if __name__ == "__main__":
    main()
