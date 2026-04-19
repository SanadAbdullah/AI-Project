"""
Interactive QA for auto-labeled images: review boxes, keep or delete pairs.

Opens each image with overlaid YOLO boxes (from labels/<class>/<stem>.txt).
Keys: K / SPACE = keep and advance; D / Backspace / Delete = remove image + label
from disk and stay at index; Q / ESC = quit current class run.

Usage:
    python clean_dataset.py              # all classes in CLASS_NAMES order
    python clean_dataset.py traffic_cone # single class name from CLASS_NAMES values
"""
import cv2
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = Path("dataset")
LABELS_DIR  = Path("labels")

# Maps YOLO class index → folder name under dataset/ and labels/
CLASS_NAMES = {
    0: "safety_helmet",
    1: "fire_extinguisher",
    2: "emergency_exit",
    3: "traffic_cone",
}

# BGR colors for rectangle/text per class id (OpenCV uses BGR)
COLORS = {
    0: (0, 255, 0),      # green
    1: (0, 0, 255),      # red
    2: (255, 165, 0),    # orange
    3: (255, 0, 255),    # magenta
}

MAX_DISPLAY_SIZE = 900  # shrink large photos so they fit typical displays

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def resize_for_display(img):
    """Uniformly scale down so max(h,w) ≤ MAX_DISPLAY_SIZE."""
    h, w = img.shape[:2]
    if max(h, w) > MAX_DISPLAY_SIZE:
        scale = MAX_DISPLAY_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def draw_boxes_on_image(img, label_path, _class_idx):
    """Overlay YOLO boxes from label file; red frame + text if missing or empty labels."""
    h, w = img.shape[:2]

    if not label_path.exists() or label_path.stat().st_size == 0:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.putText(img, "NO DETECTION", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return img

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls   = int(parts[0])
        cx    = float(parts[1])
        cy    = float(parts[2])
        bw    = float(parts[3])
        bh    = float(parts[4])
        # Normalized cx,cy,w,h → pixel axis-aligned rectangle
        x1    = int((cx - bw / 2) * w)
        y1    = int((cy - bh / 2) * h)
        x2    = int((cx + bw / 2) * w)
        y2    = int((cy + bh / 2) * h)
        color = COLORS.get(cls, (255, 255, 255))
        name  = CLASS_NAMES.get(cls, f"class_{cls}")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, name, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img


def add_hud(img, class_name, index, total, kept, deleted):
    """Bottom bar with progress and key hints."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 50), (w, h), (30, 30, 30), -1)
    text = f"{class_name}  {index+1}/{total}  kept:{kept}  deleted:{deleted}  |  K/SPACE=keep  D=delete  Q=quit"
    cv2.putText(img, text, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return img


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def clean_class(class_name, class_idx):
    """Review loop for one class; returns (kept_count, deleted_count)."""
    img_dir   = DATASET_DIR / class_name
    label_dir = LABELS_DIR  / class_name

    if not img_dir.exists():
        print(f"SKIPPING: {class_name} — folder not found")
        return 0, 0

    images = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.jpeg")) +
        list(img_dir.glob("*.png"))
    )

    if not images:
        print(f"SKIPPING: {class_name} — no images found")
        return 0, 0

    total   = len(images)
    kept    = 0
    deleted = 0

    print(f"\nStarting {class_name} — {total} images")
    print("K or SPACE = keep | D = delete | Q = quit\n")

    cv2.namedWindow("Dataset Cleaner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Cleaner", MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE)

    i = 0
    while i < len(images):
        img_path   = images[i]
        label_path = label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            i += 1
            continue

        display = draw_boxes_on_image(img.copy(), label_path, class_idx)
        display = resize_for_display(display)
        display = add_hud(display, class_name, i, total, kept, deleted)

        cv2.imshow("Dataset Cleaner", display)
        cv2.setWindowTitle("Dataset Cleaner", f"{class_name} — {i+1}/{total}")

        while True:
            key = cv2.waitKey(0) & 0xFF

            # KEEP — K, SPACE; 83/3 are OpenCV arrow-key codes on some platforms
            if key in [ord('k'), ord('K'), 32, 83, 3]:
                kept += 1
                i += 1
                break

            # DELETE — D; 8 = backspace, 127 = delete
            elif key in [ord('d'), ord('D'), 8, 127]:
                img_path.unlink(missing_ok=True)
                if label_path.exists():
                    label_path.unlink()
                deleted += 1
                images.pop(i)
                break

            # QUIT — Q, ESC
            elif key in [ord('q'), ord('Q'), 27]:
                cv2.destroyAllWindows()
                print(f"Quit. Kept: {kept}, Deleted: {deleted}")
                return kept, deleted

    cv2.destroyAllWindows()
    print(f"Finished {class_name}. Kept: {kept}, Deleted: {deleted}")
    return kept, deleted


def main():
    # Optional argv[1] filters to one folder name, e.g. python3 clean_dataset.py fire_extinguisher
    if len(sys.argv) > 1:
        target = sys.argv[1]
        for idx, name in CLASS_NAMES.items():
            if name == target:
                clean_class(name, idx)
                return
        print(f"Unknown class: {target}")
        print(f"Available: {list(CLASS_NAMES.values())}")
        return

    total_kept    = 0
    total_deleted = 0

    for class_idx, class_name in CLASS_NAMES.items():
        kept, deleted  = clean_class(class_name, class_idx)
        total_kept    += kept
        total_deleted += deleted

    print(f"\n── Final Summary ────────────────────────────")
    print(f"Total kept:    {total_kept}")
    print(f"Total deleted: {total_deleted}")
    print(f"────────────────────────────────────────────")


if __name__ == "__main__":
    main()