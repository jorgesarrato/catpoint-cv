"""
Cat tracker — dataset generation pipeline.

Usage:
    python main.py [--output data/raw] [--conf 0.4] [--no-display]
"""

import argparse
import logging
import os
import signal
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

logger = logging.getLogger(__name__)


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
    p.add_argument("--frame-skip", type=int, default=0,
                   help="Skip N frames between inferences (0 = process every frame)")
    p.add_argument("--clahe", action="store_true",
                   help="Enable CLAHE preprocessing to correct overexposure")
    p.add_argument("--clahe-clip", type=float, default=2.0,
                   help="CLAHE clip limit (default 2.0, try 3.0-4.0 for heavy overexposure)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO)")
    p.add_argument("--log-file", default=None,
                   help="Log to file in addition to stderr")
    p.add_argument("--threads", type=int, default=0,
                   help="OpenVINO inference threads (0 = auto-detect, default: 0)")
    return p.parse_args()


def main():
    args = parse_args()

    # Pin OpenVINO inference threads before model is loaded
    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    # --debug implies DEBUG log level
    if args.debug and args.log_level == "INFO":
        args.log_level = "DEBUG"

    # Configure logging
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

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

    # Graceful shutdown on SIGINT/SIGTERM (for headless --no-display runs)
    stop_requested = False

    def _signal_handler(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        logger.info("Shutdown requested (signal %d)", signum)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    stream = TapoStream().start()
    logger.info("Stream started. Press 'q' to quit, 'b' to manually save a background frame.")
    logger.info("Saving dataset to: %s", args.output)
    logger.info("CLAHE preprocessing: %s",
                f"enabled (clip={args.clahe_clip})" if args.clahe else "disabled")
    logger.info("Background saves: every %.0fs (press 'b' to save now)", args.background_interval)

    frame_count = 0
    try:
        while not stop_requested:
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            if args.frame_skip > 0 and (frame_count % (args.frame_skip + 1)) != 1:
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
                    logger.debug(items)
                else:
                    logger.debug("no detections")

            if not args.no_display:
                display_result = result if result is not None else pipeline.detector.detect(preprocessed)
                display = pipeline.detector.draw_detections(preprocessed, display_result)
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
                logger.info("Manual background frame saved")
    finally:
        stream.stop()
        time.sleep(0.2)  # let the stream thread fully exit before touching GUI
        if not args.no_display:
            cv2.destroyWindow("Cat Tracker")
            for _ in range(5):
                cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(100)
        stats = pipeline.stats
        logger.info(
            "Session summary: processed=%d detections=%d saved=%d background=%d dir=%s",
            stats['total_frames'], stats['detection_frames'],
            stats['saved_frames'], stats['background_frames'], args.output,
        )


if __name__ == "__main__":
    main()
