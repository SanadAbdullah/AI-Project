"""
Extract still frames from videos for ML dataset prep (e.g. Roboflow).

For each entry in VIDEOS, opens the video next to this script, saves every Nth frame
as a JPG under dataset/<class_name>/ with sequential filenames.
Requires: opencv-python (cv2), videos present at the paths keys in VIDEOS.
"""
import cv2
import os
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Map each video file (same folder as this script) to a label / output subfolder name
# Add or remove entries as needed
VIDEOS = {
    "IMG_8968.MOV":    "safety_helmet",
    "IMG_8967.MOV": "fire_extinguisher",
    "IMG_8969.MOV":   "emergency_exit",
    "wet_floor_sign.mov":   "wet_floor_sign",
}

# Extract 1 frame every N frames
# 24fps with EVERY_N=4 gives you 6 frames/second = ~180 frames per 30s video
EVERY_N = 4

# Output folder
OUTPUT_DIR = Path("dataset")

# ─────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────
def extract_frames(video_path, class_name, every_n, output_dir):
    """Read video_path; write one frame every every_n frames into output_dir/class_name."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return 0

    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\nProcessing: {video_path.name}")
    print(f"  FPS: {fps:.1f} | Total frames: {total_frames} | Duration: {duration:.1f}s")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Subsample: keep frames 0, every_n, 2*every_n, ... to reduce redundancy
        if frame_count % every_n == 0:
            filename = class_dir / f"{class_name}_{saved_count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"  Saved {saved_count} frames to: {class_dir}")
    return saved_count


def main():
    """Process all VIDEOS entries; skip missing files; print aggregate stats."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    total_saved = 0

    for video_file, class_name in VIDEOS.items():
        video_path = Path(video_file)

        if not video_path.exists():
            print(f"SKIPPING: {video_file} not found in project folder")
            continue

        count = extract_frames(video_path, class_name, EVERY_N, OUTPUT_DIR)
        total_saved += count

    print(f"\n── Summary ─────────────────────────────────")
    print(f"Total frames extracted: {total_saved}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print(f"────────────────────────────────────────────")
    print("\nNext step: upload the 'dataset' folder to Roboflow for annotation.")


if __name__ == "__main__":
    main()
