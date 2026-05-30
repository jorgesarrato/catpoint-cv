# catpoint-cv

A computer vision pipeline to detect and identify two cats (Salo and Taro) from a Tapo IP camera, using YOLOv8 with OpenVINO acceleration on CPU.

## Overview

The project follows an iterative three-phase workflow:

1. **Dataset collection** — stream live camera feed, detect cats with YOLO, and save diverse frames automatically
2. **Labeling** — annotate collected images in Label Studio with per-cat bounding boxes
3. **Fine-tuning & deployment** — train a cat-specific model and deploy it back to the pipeline

Each cycle improves the model's ability to distinguish Salo from Taro.

---

## Hardware Requirements

- 12+ CPUs, 32GB RAM (no GPU required for inference)
- Fine-tuning runs on Google Colab (T4 GPU) or locally on CPU (slower)
- Inference uses OpenVINO acceleration (FP32 or INT8) on CPU

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv tapoenv
source tapoenv/bin/activate
pip install -r requirements.txt

# 2. Configure camera credentials
cat > .env <<EOF
TAPO_USERNAME=your_username
TAPO_PASSWORD=your_password
TAPO_IP=192.168.x.x
EOF

# 3. Run the pipeline
python main.py --model yolov8m.pt --conf 0.25 --imgsz 1280 --clahe
```

The first run automatically exports the model to OpenVINO format (~30s one-time cost).

---

## Repository Structure

```
catpoint-cv/
├── data/
│   ├── raw/                        # pipeline output: frames + metadata JSONs
│   ├── labeled/                    # YOLO-format dataset (images + labels)
│   │   ├── images/
│   │   ├── labels/
│   │   ├── train/  val/  test/
│   │   └── dataset.yaml
│   ├── exports/                    # Label Studio export JSONs
│   ├── labelstudio_merged.json     # merged export (source of truth)
│   └── labelstudio_import.json     # import file for Label Studio
│
├── models/                         # exported OpenVINO model directories
│
├── notebooks/
│   └── train.ipynb                 # Colab fine-tuning notebook
│
├── scripts/
│   ├── export_to_labelstudio.py    # generate Label Studio import file
│   ├── merge_labelstudio_exports.py
│   ├── convert_labelstudio_export.py
│   ├── split_dataset.py
│   ├── train.py                    # fine-tune locally (CPU)
│   ├── export_openvino.py
│   └── benchmark.py               # inference latency benchmarking
│
├── src/
│   ├── dataset/
│   │   ├── pipeline.py             # orchestrates detection + filtering + saving
│   │   ├── saver.py
│   │   └── variety_filter.py       # Bhattacharyya distance deduplication
│   ├── detection/
│   │   ├── cat_detector.py         # YOLO wrapper + OpenVINO auto-export
│   │   └── preprocessor.py         # CLAHE contrast correction
│   └── stream/
│       └── tapo_stream.py          # threaded RTSP reader with auto-reconnect
│
├── tests/
├── main.py                         # entry point
└── requirements.txt
```

---

## Phase 1 — Dataset Collection

### Running the pipeline

```bash
python main.py \
    --model yolov8m.pt \
    --conf 0.25 \
    --imgsz 1280 \
    --clahe \
    --clahe-clip 3.0 \
    --display-width 1200 \
    --background-interval 60
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolov8n.pt` | Model weights (.pt) or OpenVINO directory |
| `--conf` | `0.4` | Detection confidence threshold |
| `--imgsz` | `640` | Inference resolution (use `1280` for distant/small cats) |
| `--frame-skip` | `0` | Skip N frames between inferences (reduces CPU; `1` = process every 2nd frame) |
| `--clahe` | off | Enable CLAHE contrast correction for overexposed feeds |
| `--clahe-clip` | `2.0` | CLAHE aggressiveness (try `3.0`-`4.0` for bright cameras) |
| `--similarity` | `0.15` | Bhattacharyya distance threshold for variety filter |
| `--min-interval` | `2.0` | Minimum seconds between saves |
| `--max-interval` | `30.0` | Force save after this many seconds even if cat hasn't moved |
| `--background-interval` | `60` | Seconds between automatic background frame saves |
| `--display-width` | `960` | Preview window width in pixels |
| `--no-display` | off | Disable live preview (for headless/SSH runs) |
| `--debug` | off | Log all YOLO detections per frame (all classes) |
| `--log-level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--threads` | `0` | OpenVINO inference threads (`0` = auto-detect) |
| `--log-file` | none | Additionally log to this file |

### Keyboard shortcuts

When the display window is focused:

- **q** — quit gracefully
- **b** — manually save the current frame as a background sample

For headless runs (`--no-display`), send `SIGINT` (Ctrl+C) or `SIGTERM` for a graceful shutdown with session summary.

### What gets saved

Every qualifying frame is saved to `data/raw/`:

```
20260314_103541_224_1cats.jpg             # full frame (cat detected)
20260314_103541_224_1cats_meta.json       # bounding boxes + confidence
20260314_103541_224_background.jpg        # background frame (no cat)
20260314_103541_224_background_meta.json
```

The **variety filter** prevents saving near-duplicates using HSV histogram comparison (Bhattacharyya distance). A frame is saved only if it differs visually from the last saved frame, or if the max interval has elapsed.

---

## Phase 2 — Labeling

### Export images to Label Studio

```bash
python scripts/export_to_labelstudio.py \
    --document-root /path/to/catpoint-cv \
    --skip-exported data/labelstudio_merged.json
```

This generates `data/labelstudio_import.json` containing only images not yet labeled.

### Start Label Studio

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/path/to/catpoint-cv \
label-studio start
```

### Label Studio project setup

1. Create a new project
2. Use this labeling template:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="salo"/>
    <Label value="taro"/>
    <Label value="cat"/>
  </RectangleLabels>
</View>
```

3. **Import** the generated `data/labelstudio_import.json`
4. Label each image — YOLO pre-detections are shown for cat frames; background frames are blank
5. **Export** as JSON to `data/exports/`

### Merge exports and convert to YOLO format

```bash
# Merge all export JSONs
python scripts/merge_labelstudio_exports.py --input data/exports/

# Convert to YOLO format
python scripts/convert_labelstudio_export.py \
    --export data/labelstudio_merged.json \
    --document-root /path/to/catpoint-cv
```

Output: `data/labeled/images/` and `data/labeled/labels/`.

---

## Phase 3 — Fine-tuning

### Split the dataset

```bash
python scripts/split_dataset.py --input data/labeled
```

Creates `train/`, `val/`, `test/` splits (80/10/10) and writes `data/labeled/dataset.yaml`.

### Fine-tune on Google Colab (recommended)

```bash
zip -r labeled.zip data/labeled/
```

Upload `labeled.zip` to `MyDrive/catpoint-cv/labeled.zip`, then open `notebooks/train.ipynb` in Colab:

1. **Runtime > Change runtime type > T4 GPU**
2. Set `DRIVE_ZIP_PATH` in the config cell if needed
3. **Runtime > Run all**

The best checkpoint is saved to `MyDrive/catpoint-cv/checkpoints/best.pt`.

### Fine-tune locally (CPU, slower)

```bash
python scripts/train.py --data data/labeled/dataset.yaml
```

### Export to OpenVINO

```bash
# FP32 (default)
python scripts/export_openvino.py --model best.pt --output models/

# INT8 (recommended — ~2x faster on CPU, minimal accuracy loss)
python scripts/export_openvino.py --model best.pt --int8 --data data/labeled/dataset.yaml --output models/
```

### Run inference with the fine-tuned model

```bash
# FP32
python main.py \
    --model models/best_fp32_openvino_model \
    --imgsz 1280 \
    --conf 0.25 \
    --clahe

# INT8 (faster)
python main.py \
    --model models/best_int8_openvino_model \
    --imgsz 1280 \
    --conf 0.25 \
    --clahe
```

---

## Iterative Improvement

As the fine-tuned model runs, repeat the cycle:

```
Collect more images (Phase 1)
  -> Label new images in Label Studio (Phase 2)
  -> Merge exports + convert (Phase 2)
  -> Split full dataset (Phase 3)
  -> Fine-tune from previous best.pt on ALL accumulated data (Phase 3)
  -> Export to OpenVINO (Phase 3)
  -> Deploy updated model
```

Always fine-tune from the **previous FP32 checkpoint** on the **full accumulated dataset** to avoid catastrophic forgetting.

---

## Running Tests

```bash
# Unit tests only (no network, no model download)
pytest tests/ -m "not integration" -v

# YOLO integration test (downloads yolov8n.pt + a test image)
pytest tests/ -m "integration" -v

# All tests
pytest tests/ -v
```

---

## Architecture Notes

- **TapoStream** reads RTSP frames in a background thread with automatic reconnection (exponential backoff up to 30s) on connection loss
- **CatDetector** lazy-loads the model on first inference and auto-exports `.pt` to OpenVINO; failed exports are cleaned up atomically
- **DatasetPipeline** orchestrates: preprocess (CLAHE) -> detect (YOLO) -> filter (variety) -> save
- **VarietyFilter** uses HSV histogram Bhattacharyya distance to skip near-duplicate frames, with time-based overrides
- All `print()` output uses Python's `logging` module — configure with `--log-level` and `--log-file`

### Model recommendations

Cats appear small in the frame — median crop is ~93px (max dimension) in a 1920x1080 feed. 76% of crops are under 128px. This means `imgsz=1280` is essential for reliable detection.

| Scenario | Model | imgsz | Precision | Notes |
|----------|-------|-------|-----------|-------|
| Fast collection, large cats | `yolov8n.pt` | 640 | FP32 | Lightweight, good for close-up |
| General use | `yolov8m.pt` | 1280 | FP32 | Best accuracy for small objects |
| Optimized deployment | `best_int8_openvino_model` | 1280 | INT8 | ~2x faster than FP32, <1% mAP loss |
| Maximum speed | `yolov8s` + INT8 | 1280 | INT8 | ~3-4x faster than yolov8m FP32 |

### Benchmarking

Use the benchmark script to measure actual inference latency on your hardware:

```bash
# Compare FP32 vs INT8 on a real frame
python scripts/benchmark.py \
    --models models/best_fp32_openvino_model models/best_int8_openvino_model \
    --imgsz 1280 \
    --image data/raw/20260313_161054_885_1cats.jpg

# Sweep model sizes and image resolutions
python scripts/benchmark.py \
    --models yolov8n.pt yolov8s.pt yolov8m.pt \
    --imgsz 640 1280 \
    --iterations 50

# Export results to CSV for comparison
python scripts/benchmark.py \
    --models models/best_fp32_openvino_model models/best_int8_openvino_model \
    --imgsz 1280 \
    --csv benchmark_results.csv
```

Output shows per-frame latency (mean, p50, p95, p99) and throughput in FPS:

```
Model                                         Prec     imgsz     Mean      P50      P95      P99     FPS
models/best_fp32_openvino_model               FP32      1280   120.3ms  119.1ms  125.4ms  128.7ms    8.3
models/best_int8_openvino_model               INT8      1280    62.1ms   61.5ms   65.2ms   67.8ms   16.1
```

*(Example numbers — run on your hardware for actual results.)*

### CPU performance tuning

- **INT8 quantization** is the single biggest optimization (~2x speedup over FP32)
- Use `--threads N` to pin OpenVINO inference threads (e.g., `--threads 10` to leave 2 cores for stream + OS)
- Use `--frame-skip N` to reduce CPU load (e.g., `--frame-skip 1` processes every 2nd frame)

---

## Future Features

### Real-time notifications
Send push notifications (Telegram, MQTT, or Home Assistant) when a specific cat is detected. The detection loop already identifies which cat is present — just needs a notification hook with configurable cooldown.

### Confidence-based auto-labeling
For fine-tuned model predictions with very high confidence (>0.95), auto-label new frames without manual intervention in Label Studio. Route only uncertain detections (confidence 0.3-0.7) to manual labeling, dramatically speeding up the iterative refinement loop.

### Activity tracking and logging
Track cat presence over time: when each cat appears, duration of visits, movement patterns. Store as time-series data (SQLite or InfluxDB) for visualization — e.g., "Salo was in the room from 2:00-3:30 AM".

### Multi-camera support
Extend `TapoStream` to accept multiple camera URLs. Run independent detection pipelines per camera with a shared model, or fuse detections across overlapping views.

### Web dashboard
Replace the OpenCV preview window with a lightweight web UI (FastAPI + HTMX or similar) showing live feed, detection stats, recent captures, and activity history. Enables remote monitoring without X11/display forwarding.

### Active learning pipeline
Instead of fixed-interval saving, prioritize frames where the model is most uncertain (confidence between 0.3-0.7). These are the most valuable samples for retraining and maximize improvement per labeled image.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ValueError: Missing required environment variables` | Set `TAPO_USERNAME`, `TAPO_PASSWORD`, `TAPO_IP` in `.env` |
| OpenCV window crashes on exit (Linux) | Fixed — the pipeline flushes GUI events before exit |
| Model loads slowly on first run | One-time OpenVINO export; subsequent runs load instantly |
| Corrupted `*_openvino_model/` directory | Delete it — the pipeline will re-export cleanly on next run |
| Stream silently stops receiving frames | Auto-reconnect handles this; check `--log-level DEBUG` for details |
| Too many similar frames saved | Lower `--similarity` (e.g., `0.10`) or increase `--min-interval` |
| Cats too small to detect | Use `--imgsz 1280` and a larger model (`yolov8m.pt`) |
