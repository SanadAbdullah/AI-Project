# AI-Project — Safety object detection (YOLOv8)

This repository builds a **custom object-detection dataset** for workplace safety items, uses **Grounding DINO** to propose bounding boxes from text prompts, lets you **review and clean** labels interactively, **splits** data into YOLO format, and **trains** a **YOLOv8** model with **Ultralytics**. Optional utilities help **collect**, **rename**, **convert**, and **sample** images from video.

---

## What it detects

The default configuration targets **four classes** (indices `0`–`3`), kept consistent across labeling, splitting, and training:

| Index | Class               |
|-------|---------------------|
| 0     | `safety_helmet`     |
| 1     | `fire_extinguisher` |
| 2     | `emergency_exit`    |
| 3     | `traffic_cone`      |

These names appear in `split_dataset.py`, `auto_label.py`, `clean_dataset.py`, `preview_labels.py`, and `yolo_dataset/data.yaml`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `dataset/` | Raw images grouped by class subfolders (e.g. `dataset/traffic_cone/`). |
| `labels/` | YOLO `.txt` labels mirroring `dataset/` (`labels/<class>/<image_stem>.txt`). |
| `yolo_dataset/` | Flat YOLO layout produced by `split_dataset.py`: `train/`, `val/`, `test/`, each with `images/` and `labels/`, plus `data.yaml`. |
| `testdata/` | Optional images for final **inference demo** (`train.py`). |
| `runs/` | Ultralytics outputs: training, validation plots, inference runs. |
| `useful scripts/` | Standalone helpers (downloads, rename, HEIC, frame extraction). |
| `GroundingDINO_SwinT_OGC.py`, `groundingdino_swint_ogc.pth` | Grounding DINO config/weights used by `auto_label.py` (weights may also be downloaded automatically). |
| `yolov8m.pt` | Pretrained YOLOv8 weights used when starting training. |
| `.venv/` | Python virtual environment (local; not required to be committed). |

Other folders (e.g. `datasets/`, `videos & images used before cleaning/`) are project-specific asset storage.

---

## End-to-end pipeline

Recommended order:

1. **Gather images**  
   Put images under `dataset/<class_name>/`. Use utilities in `useful scripts/` if needed (downloads, renaming, HEIC conversion, frames from video).

2. **Auto-label** — `auto_label.py`  
   Loads Grounding DINO and, for each class, runs a **text prompt** over that class’s images. Writes one YOLO label file per image under `labels/<class>/` (normalized `cx cy w h`; **empty file** = no detection).  
   **Important:** Class IDs in `CLASS_INDICES` must match `data.yaml` / `split_dataset.py`.

3. **Preview** — `preview_labels.py`  
   Samples random images per class, draws boxes (or highlights missing labels), saves JPEGs under `label_preview/` for a quick sanity check **without** the GUI.

4. **Clean** — `clean_dataset.py`  
   Interactive OpenCV viewer: **keep** or **delete** image + label pairs. Optional: `python clean_dataset.py <class_name>` to process one class.

5. **Split** — `split_dataset.py`  
   Copies only images that have a **non-empty** label into `yolo_dataset/` with a **70% / 20% / 10%** train/val/test split (seeded), and writes `yolo_dataset/data.yaml`.  
   Run: `python split_dataset.py` (script runs at **import time**).

6. **Train & evaluate** — `train.py`  
   Trains YOLOv8 if `best.pt` is not already present at the configured path, runs **validation on the test split**, shows training plots if present, and optionally runs **prediction** on `testdata/`.

---

## Main scripts (project root)

| Script | Role |
|--------|------|
| `auto_label.py` | Grounding DINO → YOLO labels in `labels/`. |
| `preview_labels.py` | Random visual previews to `label_preview/`. |
| `clean_dataset.py` | Keyboard-driven QA; removes bad pairs from `dataset/` and `labels/`. |
| `split_dataset.py` | Builds `yolo_dataset/` + `data.yaml` from `dataset/` + `labels/`. |
| `train.py` | YOLOv8 training, test metrics, plots, optional inference on `testdata/`. |

---

## Useful scripts (`useful scripts/`)

| File | Role |
|------|------|
| `bulk_image_downloader.py` | Download images from the **Pixabay** API (keyword, count, folder). |
| `bulk_image_downloaderr.py` | Same idea for the **Pexels** API. |
| `rename_files.py` | Rename photos next to the script to random 10-character names. |
| `rename_files 2.py` | Rename to `YYYY-MM-DD_NNN` using file modification time. |
| `convert_heic.py` | Recursively convert `.HEIC` → `.jpg` (deletes HEIC after success). |
| `extract_frames.py` | OpenCV: extract every *N*th frame from listed videos into `dataset/<class>/` for ML prep. |

Store **API keys in environment variables** or a local untracked config—not in committed source.

---

## Environment and dependencies

Use a **Python 3** virtual environment (e.g. `.venv` in this project). Install packages as needed for the parts you use, for example:

- **Training / inference:** `ultralytics`, `torch`, `matplotlib`
- **OpenCV UIs:** `opencv-python`
- **Auto-labeling:** `torch`, Grounding DINO stack (see the [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) project for install notes), plus any deps your `groundingdino` import expects
- **Bulk downloaders:** `requests`
- **HEIC conversion:** `pillow`, `pillow-heif`
- **Frame extraction:** `opencv-python`

**Device:** Scripts default to **Apple Silicon MPS** where noted (`train.py`, `auto_label.py`). Switch to `cuda` or `cpu` in those files if your machine differs.

---

## Configuration highlights

- **Splits** — `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`, and `SEED` live in `split_dataset.py`.
- **Training** — Epochs, image size, batch size, model name, and paths to `data.yaml` and weights are at the top of `train.py`.
- **Grounding DINO thresholds** — `BOX_THRESHOLD` and `TEXT_THRESHOLD` in `auto_label.py` control detection sensitivity.

After changing class names or counts, update **every** shared mapping (`auto_label`, `split_dataset`, preview/clean tools) and **regenerate** `data.yaml` by re-running `split_dataset.py`.

---

## Outputs to inspect

- **`yolo_dataset/data.yaml`** — Dataset definition for Ultralytics.
- **`runs/train/...`** and **`runs/detect/...`** — Training and validation artifacts (curves, confusion matrix, weights).
- **`runs/inference/demo/`** — Saved predictions when `testdata/` is used from `train.py`.

---

## License and third-party assets

Model weights (Grounding DINO, YOLOv8) and API-sourced images are subject to their respective **licenses and terms**. Follow Pixabay/Pexels/API rules when downloading.
