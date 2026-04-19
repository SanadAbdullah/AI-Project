"""
Build a flat YOLO dataset layout from per-class folders.

Reads dataset/<class>/*.jpg|jpeg|png and matching labels/<class>/<stem>.txt.
Only copies images whose label file exists and is non-empty (same rule as training quality).
Shuffles per class with SEED, splits 70/20/10 into yolo_dataset/{train,val,test}/{images,labels},
then writes yolo_dataset/data.yaml for Ultralytics.

Run once:  python split_dataset.py
(This file runs the split at import time — no main() guard.)
"""
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = Path("dataset")
LABELS_DIR  = Path("labels")
OUTPUT_DIR  = Path("yolo_dataset")

# Order defines class indices 0..n-1 in data.yaml (must match labels’ numeric ids)
CLASS_NAMES = [
    "safety_helmet",
    "fire_extinguisher",
    "emergency_exit",
    "traffic_cone",
]

# Must sum to 1.0; remainder after train/val slices goes to test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

SEED = 42

# ─────────────────────────────────────────────
# CREATE FOLDER STRUCTURE
# ─────────────────────────────────────────────

for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# SPLIT AND COPY
# ─────────────────────────────────────────────

random.seed(SEED)
summary = {}

for class_name in CLASS_NAMES:
    img_dir   = DATASET_DIR / class_name
    label_dir = LABELS_DIR  / class_name

    if not img_dir.exists():
        print(f"SKIPPING: {class_name} — folder not found")
        continue

    # Skip unlabeled or failed auto-label runs (empty .txt)
    images = []
    for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")):
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists() and label_path.stat().st_size > 0:
            images.append(img_path)

    if not images:
        print(f"SKIPPING: {class_name} — no labeled images found")
        continue

    random.shuffle(images)

    n       = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    # Assign leftover images to test so train+val+test == n when ratios don’t divide evenly
    n_test  = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        for img_path in split_images:
            label_path = label_dir / (img_path.stem + ".txt")

            shutil.copy(img_path, OUTPUT_DIR / split_name / "images" / img_path.name)

            # Parallel filename in labels/ (already verified non-empty for this image)
            if label_path.exists():
                shutil.copy(label_path, OUTPUT_DIR / split_name / "labels" / label_path.name)

    summary[class_name] = {"total": n, "train": n_train, "val": n_val, "test": n_test}
    print(f"{class_name}: {n} total → train:{n_train}  val:{n_val}  test:{n_test}")

# ─────────────────────────────────────────────
# WRITE data.yaml
# ─────────────────────────────────────────────
# Ultralytics expects path + relative splits; names list defines class order

yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

yaml_path = OUTPUT_DIR / "data.yaml"
yaml_path.write_text(yaml_content)

print(f"\n── Summary ─────────────────────────────────")
total_images = sum(v["total"] for v in summary.values())
print(f"Total images used: {total_images}")
print(f"data.yaml written to: {yaml_path.resolve()}")
print(f"Dataset ready at: {OUTPUT_DIR.resolve()}")
print("────────────────────────────────────────────")
print("\nNext step: run python3 train.py")
