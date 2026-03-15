"""
Cat tracker — dataset generation pipeline.

Usage:
    python main.py [--output data/raw] [--conf 0.4] [--no-display]
"""

import argparse
import os
import time
import cv2
from dotenv import load_dotenv

from src.stream.tapo_stream import TapoStream
from src.detection.cat_detector import CatDetector
from src.detection.preprocessor import CLAHEPreprocessor
from src.dataset.variety_filter import VarietyFilter
from src.dataset.saver import DatasetSaver
from src.dataset.pipeline import DatasetPipeline

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Cat dataset generation pipeline")
    p.add_argument("--output", default="data/raw", help="Output directory for dataset")
    p.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO inference resolution (default 640, try 1280 for small objects)")
    p.add_argument("--similarity", type=float, default=0.15,
                   help="Bhattacharyya distance threshold for variety filter")
    p.add_argument("--min-interval", type=float, default=2.0,
                   help="Min seconds between saves")
    p.add_argument("--max-interval", type=float, default=30.0,
                   help="Max seconds before forcing a save (even if cat hasn't moved)")
    p.add_argument("--background-interval", type=float, default=60.0,
                   help="Seconds between automatic background saves (default 60)")
    p.add_argument("--no-display", action="store_true", help="Disable live preview window")
    p.add_argument("--display-width", type=int, default=960,
                   help="Preview window width in pixels (default 960)")
    p.add_argument("--debug", action="store_true",
                   help="Print all YOLO detections per frame (ignores class filter)")
    p.add_argument("--clahe", action="store_true",
                   help="Enable CLAHE preprocessing to correct overexposure")
    p.add_argument("--clahe-clip", type=float, default=2.0,
                   help="CLAHE clip limit (default 2.0, try 3.0-4.0 for heavy overexposure)")
    return p.parse_args()


def main():
    args = parse_args()

    pipeline = DatasetPipeline(
        detector=CatDetector(model_path=args.model, confidence_threshold=args.conf, imgsz=args.imgsz),
        variety_filter=VarietyFilter(
            similarity_threshold=args.similarity,
            min_interval_sec=args.min_interval,
            max_interval_sec=args.max_interval,
            background_interval_sec=args.background_interval,
        ),
        saver=DatasetSaver(output_dir=args.output),
        preprocessor=CLAHEPreprocessor(clip_limit=args.clahe_clip, enabled=args.clahe),
    )

    stream = TapoStream().start()
    print("Stream started. Press 'q' to quit, 'b' to manually save a background frame.")
    print(f"Saving dataset to: {args.output}")
    print(f"CLAHE preprocessing: {'enabled (clip={})'.format(args.clahe_clip) if args.clahe else 'disabled'}")
    print(f"Background saves: every {args.background_interval:.0f}s (press 'b' to save now)")

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            preprocessed, result = pipeline.process(frame)

            if args.debug:
                all_detections = pipeline.detector.detect_all(preprocessed)
                if all_detections:
                    items = ", ".join(
                        f"{name}: {conf:.2f}" for name, conf in sorted(
                            all_detections.items(), key=lambda x: -x[1]
                        )
                    )
                    print(f"[DEBUG] {items}")
                else:
                    print("[DEBUG] no detections")

            if not args.no_display:
                display = preprocessed.copy()
                if result is not None:
                    display = pipeline.detector.draw_detections(display, result)
                stats = pipeline.stats
                text = (f"Saved: {stats['saved_frames']} | "
                        f"Detected: {stats['detection_frames']} | "
                        f"Background: {stats['background_frames']}")
                cv2.putText(display, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                # Resize to fit screen while preserving aspect ratio
                h, w = display.shape[:2]
                target_w = args.display_width
                target_h = int(h * target_w / w)
                display = cv2.resize(display, (target_w, target_h))
                cv2.imshow("Cat Tracker", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                pipeline.saver.save_background(preprocessed)
                pipeline.variety_filter.reset_background_timer()
                print(f"[MANUAL] Background frame saved")
    finally:
        stream.stop()
        time.sleep(0.2)  # let the stream thread fully exit before touching GUI
        if not args.no_display:
            # Explicitly destroy by name first, then flush the event queue
            # multiple times — the single waitKey(1) is not always enough
            # to drain pending native GUI callbacks on Linux, causing a segfault.
            cv2.destroyWindow("Cat Tracker")
            for _ in range(5):
                cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(100)
        stats = pipeline.stats
        print("\n--- Session Summary ---")
        print(f"  Frames processed : {stats['total_frames']}")
        print(f"  Cat detections   : {stats['detection_frames']}")
        print(f"  Images saved     : {stats['saved_frames']}")
        print(f"  Background saved : {stats['background_frames']}")
        print(f"  Output dir       : {args.output}")


if __name__ == "__main__":
    main()
