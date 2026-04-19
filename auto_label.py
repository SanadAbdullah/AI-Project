"""
Generate YOLO-format label files using Grounding DINO (open-vocabulary detection).

For each class folder under dataset/<class>/, runs the model with that class's text
prompt and writes labels/<class>/<stem>.txt with normalized cx cy w h per line
(YOLO bbox format, values in [0,1]). Class IDs must match split_dataset / data.yaml.
Requires: torch, supervision, groundingdino (and model weights on disk or downloaded).
"""
import torch
from groundingdino.util.inference import load_model, load_image, predict
from pathlib import Path
import urllib.request

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Open-vocab phrases sent to Grounding DINO (one caption per dataset class)
# Remove or comment out entries to skip relabeling that class folder
CLASS_PROMPTS = {
    "safety_helmet":     "safety helmet",
    "fire_extinguisher": "fire extinguisher",
    "emergency_exit":    "emergency exit sign",
    "traffic_cone":      "traffic cone",
}

# YOLO class IDs — must stay in lockstep with split_dataset CLASS_NAMES / data.yaml
CLASS_INDICES = {
    "safety_helmet":     0,
    "fire_extinguisher": 1,
    "emergency_exit":    2,
    "traffic_cone":      3,
}

# Images live under dataset/<class_name>/
DATASET_DIR = Path("dataset")

# Writes parallel tree: labels/<class_name>/<image_stem>.txt
LABELS_DIR = Path("labels")

# Higher = fewer but more confident boxes; tune for precision vs recall
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.25

# Model config and weights
CONFIG_PATH  = "GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
WEIGHTS_URL  = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
CONFIG_URL   = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# ─────────────────────────────────────────────
# DOWNLOAD MODEL IF NEEDED
# ─────────────────────────────────────────────
def download_if_missing(path, url):
    """Fetch config or weights from url if path is not present locally."""
    if not Path(path).exists():
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {path}")

# ─────────────────────────────────────────────
# AUTO LABEL
# ─────────────────────────────────────────────
def auto_label():
    """Walk each class, run detection, emit one .txt per image (empty if no boxes)."""
    download_if_missing(CONFIG_PATH, CONFIG_URL)
    download_if_missing(WEIGHTS_PATH, WEIGHTS_URL)

    print("Loading Grounding DINO model...")
    # Apple Silicon GPU when available; GroundingDINO may also support CUDA if you change this
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=device)
    print(f"Model loaded on {device}")

    total_labeled = 0
    total_skipped = 0

    for class_name, prompt in CLASS_PROMPTS.items():
        class_idx = CLASS_INDICES[class_name]
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"\nSKIPPING: {class_name} folder not found")
            continue

        label_dir = LABELS_DIR / class_name
        label_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.glob("*.jpg")) + \
                 list(class_dir.glob("*.jpeg")) + \
                 list(class_dir.glob("*.png"))

        print(f"\n── {class_name} (index={class_idx}) — {len(images)} images ──────────────────")

        for img_path in images:
            try:
                image_source, image_tensor = load_image(str(img_path))

                # predict returns boxes already in normalized center-x, center-y, w, h (YOLO-style)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image_tensor,
                    caption=prompt,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device=device,
                )

                label_path = label_dir / (img_path.stem + ".txt")

                if len(boxes) == 0:
                    # Empty file marks “no object” for downstream filtering / preview
                    label_path.write_text("")
                    total_skipped += 1
                    print(f"  ✗ No detection: {img_path.name}")
                    continue

                with open(label_path, "w") as f:
                    for box in boxes:
                        cx, cy, bw, bh = box.tolist()
                        # One line per instance: class_id cx cy w h (all normalized 0–1)
                        f.write(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                total_labeled += 1
                print(f"  ✓ {img_path.name} — {len(boxes)} box(es) | class_idx={class_idx}")

            except Exception as e:
                print(f"  ✗ Error on {img_path.name}: {e}")
                total_skipped += 1

    print(f"\n── Summary ─────────────────────────────────")
    print(f"Successfully labeled: {total_labeled}")
    print(f"No detection / skipped: {total_skipped}")
    print(f"Labels saved to: {LABELS_DIR.resolve()}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    auto_label()