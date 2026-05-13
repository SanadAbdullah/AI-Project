"""
End-to-end YOLOv8 training script: train (if needed), test-set validation, plot display, demo infer.

- If best weights already exist at BEST_WEIGHTS, skips training and reuses them.
- Validates on the test split from data.yaml and prints box metrics.
- Optionally displays training curves from the Ultralytics run folder.
- Runs prediction on TEST_DIR when it contains images; saves under runs/inference/demo.

Requires: ultralytics, torch, matplotlib; expects split_dataset.py output layout + paths below.
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_YAML  = Path("yolo_dataset/data.yaml")
BEST_WEIGHTS  = Path("runs/detect/runs/train/safety-detection-v2/weights/best.pt")

MODEL_SIZE    = "yolov8m.pt"
EPOCHS        = 100
IMAGE_SIZE    = 640
BATCH_SIZE    = 20
DEVICE        = "mps"
FORCE_RETRAIN = False

TEST_DIR      = Path("testdata")

# ─────────────────────────────────────────────
# 1. TRAIN (skip if already trained)
# ─────────────────────────────────────────────

print("Starting training...")
print(f"Dataset:  {DATASET_YAML.resolve()}")
print(f"Device:   {DEVICE}")
print()

if not FORCE_RETRAIN and BEST_WEIGHTS.exists():
    print(f"Using existing weights: {BEST_WEIGHTS}")
else:
    last_weights = Path("runs/detect/runs/train/safety-detection-v2/weights/last.pt")
    if last_weights.exists():
        print(f"Resuming training from: {last_weights}")
        model = YOLO(str(last_weights))
        model.train(resume=True)
    else:
        print(f"Starting fresh with {MODEL_SIZE}")
        model = YOLO(MODEL_SIZE)
        model.train(
            data=str(DATASET_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project="runs/train",
            name="safety-detection-v2",
            patience=30,
            save=True,
            plots=True,
            verbose=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.1,
            mosaic=1.0,
            mixup=0.1,
        )

print(f"\nBest weights: {BEST_WEIGHTS}")

# ─────────────────────────────────────────────
# 2. EVALUATE ON TEST SET
# ─────────────────────────────────────────────

print("\nRunning evaluation on test set...")
best_model = YOLO(str(BEST_WEIGHTS))

metrics = best_model.val(
    data=str(DATASET_YAML),
    split="test",
    device=DEVICE,
    plots=True,
    save_json=True,
)

print("\n── Evaluation Results ──────────────────────")
print(f"Precision:      {metrics.box.mp:.4f}")
print(f"Recall:         {metrics.box.mr:.4f}")
print(f"mAP@0.5:        {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:   {metrics.box.map:.4f}")
print("────────────────────────────────────────────")

# ─────────────────────────────────────────────
# 3. SHOW TRAINING PLOTS
# ─────────────────────────────────────────────

plots_dir = Path("runs/detect/runs/train/safety-detection-v2")
plot_files = [
    "confusion_matrix.png",
    "results.png",
    "PR_curve.png",
    "F1_curve.png",
]

print("\nDisplaying training plots...")
for plot_file in plot_files:
    plot_path = plots_dir / plot_file
    if plot_path.exists():
        img = mpimg.imread(str(plot_path))
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.title(plot_file.replace(".png", "").replace("_", " "))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(plots_dir / f"display_{plot_file}"))
        plt.show()
    else:
        print(f"Plot not found: {plot_path}")

# ─────────────────────────────────────────────
# 4. INFERENCE ON TESTDATA FOLDER (LIVE DEMO)
# ─────────────────────────────────────────────

if TEST_DIR.exists() and any(TEST_DIR.iterdir()):
    images = list(TEST_DIR.glob("*.jpg")) + \
             list(TEST_DIR.glob("*.jpeg")) + \
             list(TEST_DIR.glob("*.png"))

    if images:
        print(f"\nRunning inference on {len(images)} image(s) in '{TEST_DIR}/'...")
        inference_results = best_model.predict(
            source=str(TEST_DIR),
            device=DEVICE,
            conf=0.25,
            save=True,
            save_txt=True,
            project="runs/inference",
            name="demo",
        )

        for r in inference_results:
            img_name = Path(r.path).name
            boxes = r.boxes
            print(f"\n{img_name} — {len(boxes)} detection(s):")
            if len(boxes) == 0:
                print("  No objects detected.")
            for box in boxes:
                cls_id   = int(box.cls[0])
                cls_name = best_model.names[cls_id]
                conf     = float(box.conf[0])
                print(f"  Class: {cls_name:<25} Confidence: {conf:.2%}")

        print(f"\nInference results saved to: runs/inference/demo/")
    else:
        print(f"\nNo images found in '{TEST_DIR}/'. Add .jpg or .png files and rerun.")
else:
    print(f"\n'{TEST_DIR}/' folder not found. Create it, add test images, and rerun.")

print("\nAll done. Check the runs/ folder for all outputs.")