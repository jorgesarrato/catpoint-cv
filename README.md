# catpoint-cv

A computer vision pipeline to detect and identify two cats (Salo and Taro) from a Tapo IP camera, using YOLOv8 with OpenVINO acceleration.

## Overview

The pipeline has two phases:

1. **Dataset generation** — run the camera feed through a pretrained YOLO model to collect labeled images of your cats
2. **Fine-tuning** — label the images in Label Studio and fine-tune the model to distinguish between the two cats

---

## Hardware

- 12 CPUs, 32GB RAM (no GPU)
- Fine-tuning runs on Google Colab (T4 GPU)
- Inference runs locally with OpenVINO acceleration on CPU

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
│   └── export_openvino.py
│
├── src/
│   ├── dataset/
│   │   ├── pipeline.py             # orchestrates detection + saving
│   │   ├── saver.py
│   │   └── variety_filter.py
│   ├── detection/
│   │   ├── cat_detector.py         # YOLO wrapper + OpenVINO export
│   │   └── preprocessor.py         # CLAHE preprocessing
│   └── stream/
│       └── tapo_stream.py          # threaded RTSP stream reader
│
├── tests/
├── main.py
└── requirements.txt
```

---

## Installation

```bash
python -m venv tapoenv
source tapoenv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
TAPO_USERNAME=your_username
TAPO_PASSWORD=your_password
TAPO_IP=192.168.x.x
```

---

## Phase 1 — Dataset Collection

### Run the pipeline

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

The first run will automatically export `yolov8m.pt` to OpenVINO FP32 format (takes ~30s, runs faster on subsequent launches).

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | Model weights or OpenVINO directory |
| `--conf` | `0.4` | YOLO confidence threshold |
| `--imgsz` | `640` | Inference resolution (use `1280` for small cats) |
| `--clahe` | off | Enable CLAHE contrast correction for overexposed feeds |
| `--clahe-clip` | `2.0` | CLAHE aggressiveness (try `3.0`–`4.0` for bright cameras) |
| `--display-width` | `960` | Preview window width in pixels |
| `--background-interval` | `60` | Seconds between automatic background frame saves |
| `--no-display` | off | Disable live preview (useful for headless runs) |
| `--debug` | off | Print all YOLO detections per frame |

**Keyboard shortcuts** (when display window is focused):

- `q` — quit
- `b` — manually save the current frame as a background sample (resets the automatic background timer)

### What gets saved

Every qualifying frame is saved to `data/raw/`:

```
20260314_103541_224_1cats.jpg       full frame (cat detected)
20260314_103541_224_1cats_meta.json bounding boxes + confidence
20260314_103541_224_background.jpg  background frame (no cat)
20260314_103541_224_background_meta.json
```

The **variety filter** prevents saving near-duplicate frames — a frame is only saved if it is visually different enough from the last saved one, or if a minimum time interval has elapsed.

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

Or add those variables to `~/.local/share/label-studio/.env` and use your `run-label-studio.sh`.

### Label Studio setup

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

3. **Import** → upload `data/labelstudio_import.json`
4. Label each image — bounding boxes from YOLO are pre-loaded for cat frames; background frames are blank for manual annotation
5. **Export** → JSON format → save to `data/exports/`

### Merge exports and convert to YOLO format

After labeling sessions, merge all export files and convert:

```bash
# Merge all export JSONs in data/exports/
python scripts/merge_labelstudio_exports.py --input data/exports/

# Convert to YOLO format
python scripts/convert_labelstudio_export.py \
    --export data/labelstudio_merged.json \
    --document-root /path/to/catpoint-cv
```

Output lands in `data/labeled/images/` and `data/labeled/labels/`.

---

## Phase 3 — Fine-tuning

### Split the dataset

```bash
python scripts/split_dataset.py --input data/labeled
```

This creates `train/`, `val/`, `test/` splits (80/10/10) and writes `data/labeled/dataset.yaml`.

### Fine-tune on Google Colab

```bash
# Zip the dataset
zip -r labeled.zip data/labeled/
```

Upload `labeled.zip` to `MyDrive/catpoint-cv/labeled.zip`, then open `notebooks/train.ipynb` in Colab:

1. **Runtime → Change runtime type → T4 GPU**
2. Set `DRIVE_ZIP_PATH` in the config cell if needed
3. **Runtime → Run all**

`best.pt` is saved automatically to `MyDrive/catpoint-cv/checkpoints/best.pt`.

### Fine-tune locally (CPU, slower)

```bash
python scripts/train.py --data data/labeled/dataset.yaml
```

### Export to OpenVINO

```bash
python scripts/export_openvino.py --model best.pt --output models/
```

### Run inference with fine-tuned model

```bash
python main.py \
    --model models/best_openvino_model \
    --imgsz 1280 \
    --conf 0.25 \
    --clahe
```

---

## Iterative improvement

As the fine-tuned model collects more images, repeat the labeling and fine-tuning cycle:

```
collect more images (Phase 1)
    → label new images in Label Studio (Phase 2)
    → merge exports + convert (Phase 2)
    → split dataset (Phase 3)
    → fine-tune from previous best.pt checkpoint (Phase 3)
    → export to OpenVINO (Phase 3)
    → deploy
```

Always fine-tune from the **previous FP32 checkpoint** on the **full accumulated dataset** — not just new images, to avoid catastrophic forgetting.

---

## Running tests

```bash
# Unit tests only (no network, no model download)
pytest tests/ -m "not integration" -v

# YOLO integration test (downloads yolov8n.pt and a test image)
pytest tests/ -m "integration" -v
```

---

## Model notes

- **YOLOv8m** is recommended — good balance of accuracy and CPU speed
- **imgsz=1280** is recommended for cats occupying ~100px in a 1920×1080 frame
- **CLAHE** helps with overexposed or IR night-mode frames
- OpenVINO FP32 gives ~3–6x speedup over raw PyTorch on CPU
- The model is exported automatically on first run — delete the `*_openvino_model/` directory to force a re-export
