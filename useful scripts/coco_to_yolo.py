"""
Convert COCO-format annotations to YOLO .txt labels and copy images
into dataset/<class_name>/ so they flow through your existing pipeline.

Usage:
  python coco_to_yolo.py \
    --coco_json path/to/annotations.json \
    --images_dir path/to/coco_images/ \
    --output_dataset dataset/

Class name mapping must match your CLASS_INDICES exactly.
"""

import json
import shutil
from pathlib import Path
import argparse

# Must match auto_label.py / split_dataset.py
CLASS_NAME_MAP = {
    # Emergency exit subtypes → all map to emergency_exit
    "left exit": "emergency_exit",
    "left/right exit": "emergency_exit",
    "right exit": "emergency_exit",
    "straight exit": "emergency_exit",
    "rexit-lexit-sexit-bexit-lrexit": "emergency_exit",
    # Other classes (for other datasets)
    "helmet": "safety_helmet",
    "safety helmet": "safety_helmet",
    "fire extinguisher": "fire_extinguisher",
    "traffic cone": "traffic_cone",
    "cone": "traffic_cone",
}

def convert(coco_json: Path, images_dir: Path, output_dataset: Path):
    with open(coco_json) as f:
        coco = json.load(f)

    # Build lookup dicts
    cat_id_to_name = {c["id"]: c["name"].lower() for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    from collections import defaultdict
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    converted = skipped_cat = skipped_img = 0

    for img_id, anns in ann_by_image.items():
        img_info = img_id_to_info.get(img_id)
        if not img_info:
            skipped_img += 1
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]
        img_file = images_dir / img_info["file_name"]

        if not img_file.exists():
            print(f"  Missing image: {img_file}")
            skipped_img += 1
            continue

        yolo_lines = []
        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], "").lower()
            class_name = CLASS_NAME_MAP.get(cat_name)
            if not class_name:
                skipped_cat += 1
                continue

            class_idx = {"safety_helmet":0,"fire_extinguisher":1,"emergency_exit":2,"traffic_cone":3}[class_name]
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            yolo_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Copy image to dataset/<class_name>/
            dest_dir = output_dataset / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_img = dest_dir / img_file.name
            if not dest_img.exists():
                shutil.copy2(img_file, dest_img)

        if yolo_lines:
            # Write label directly — skip auto_label for these (already annotated)
            from pathlib import Path as P
            label_dir = P("labels") / class_name
            label_dir.mkdir(parents=True, exist_ok=True)
            label_file = label_dir / (img_file.stem + ".txt")
            label_file.write_text("\n".join(yolo_lines) + "\n")
            converted += 1

    print(f"Converted: {converted} images")
    print(f"Skipped (missing image): {skipped_img}")
    print(f"Skipped (unknown category): {skipped_cat}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco_json", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--output_dataset", default="dataset")
    args = p.parse_args()
    convert(Path(args.coco_json), Path(args.images_dir), Path(args.output_dataset))