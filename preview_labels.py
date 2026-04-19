"""
Quick visual audit of labels without opening the GUI cleaner.

Randomly samples SAMPLES_PER_CLASS images per class, draws boxes (or a red border
when the label file is missing/empty), and writes JPEGs under label_preview/<class>/.

Use this after auto_label.py to spot-check quality before clean_dataset / split_dataset.
"""
import cv2
import random
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = Path("dataset")
LABELS_DIR  = Path("labels")
OUTPUT_DIR  = Path("label_preview")

# Same id→name mapping as clean_dataset / split_dataset
CLASS_NAMES = {
    0: "safety_helmet",
    1: "fire_extinguisher",
    2: "emergency_exit",
    3: "traffic_cone",
}

COLORS = {
    0: (0, 255, 0),    # green
    1: (0, 0, 255),    # red
    2: (255, 165, 0),  # orange
    3: (255, 0, 255),  # magenta
}

# How many random images to preview per class
SAMPLES_PER_CLASS = 5

# ─────────────────────────────────────────────
# DRAW BOXES
# ─────────────────────────────────────────────
def draw_boxes(img_path, label_path, _class_idx):
    """Load image and overlay YOLO boxes from label_path (_class_idx reserved for callers)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    if not label_path.exists() or label_path.stat().st_size == 0:
        # No detection — draw red border to flag it
        cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 0, 255), 5)
        cv2.putText(img, "NO DETECTION", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return img

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # YOLO normalized coords → pixels (same math as clean_dataset.draw_boxes_on_image)
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        color = COLORS.get(cls, (255, 255, 255))
        label = CLASS_NAMES.get(cls, f"class_{cls}")

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img


def main():
    """Render random previews per class and print counts of empty-label samples."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    total_no_detection = 0
    total_checked = 0

    for class_idx, class_name in CLASS_NAMES.items():
        img_dir   = DATASET_DIR / class_name
        label_dir = LABELS_DIR  / class_name

        if not img_dir.exists():
            print(f"SKIPPING: {class_name} not found")
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if not images:
            print(f"SKIPPING: no images in {class_name}")
            continue

        # Pick random samples
        samples = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

        class_output = OUTPUT_DIR / class_name
        class_output.mkdir(exist_ok=True)

        no_det = 0
        for img_path in samples:
            label_path = label_dir / (img_path.stem + ".txt")
            result = draw_boxes(img_path, label_path, class_idx)

            if result is not None:
                out_path = class_output / img_path.name
                cv2.imwrite(str(out_path), result)

                if not label_path.exists() or label_path.stat().st_size == 0:
                    no_det += 1

            total_checked += 1

        total_no_detection += no_det
        print(f"{class_name}: {len(samples)} previewed, {no_det} with no detection")

    print(f"\n── Summary ─────────────────────────────────")
    print(f"Total previewed:     {total_checked}")
    print(f"No detection:        {total_no_detection}")
    print(f"Preview saved to:    {OUTPUT_DIR.resolve()}")
    print("────────────────────────────────────────────")
    print("\nOpen the 'label_preview' folder to visually inspect the results.")


if __name__ == "__main__":
    main()
