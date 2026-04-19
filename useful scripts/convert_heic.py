"""
Batch-convert Apple HEIC/HEIF images to JPEG using Pillow + pillow-heif.

Walks INPUT_DIR recursively (*.HEIC / *.heic), saves a .jpg next to each file,
then deletes the original HEIC on success. Requires: pillow, pillow-heif.
"""
import os
from pathlib import Path
from PIL import Image
import pillow_heif

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Root to scan (defaults to current working directory when you run the script)
INPUT_DIR = Path(".")

# ─────────────────────────────────────────────
# CONVERSION
# ─────────────────────────────────────────────
def convert_heic_to_jpg(input_dir):
    # Teach Pillow how to open HEIF/HEIC via pillow_heif
    pillow_heif.register_heif_opener()

    # Merge both casings; rglob is case-sensitive on some filesystems
    heic_files = list(input_dir.rglob("*.HEIC")) + list(input_dir.rglob("*.heic"))

    if not heic_files:
        print("No HEIC files found.")
        return

    print(f"Found {len(heic_files)} HEIC files. Converting...")

    converted = 0
    failed = 0

    for heic_path in heic_files:
        jpg_path = heic_path.with_suffix(".jpg")
        try:
            img = Image.open(heic_path)
            # HEIC may be extended range; RGB is safe for standard JPEG
            img.convert("RGB").save(jpg_path, "JPEG", quality=95)
            os.remove(heic_path)  # only removed after successful write
            print(f"  ✓ {heic_path.name} → {jpg_path.name}")
            converted += 1
        except Exception as e:
            print(f"  ✗ Failed: {heic_path.name} — {e}")
            failed += 1

    print(f"\n── Summary ─────────────────────────────────")
    print(f"Converted: {converted}")
    print(f"Failed:    {failed}")
    print(f"────────────────────────────────────────────")


if __name__ == "__main__":
    convert_heic_to_jpg(INPUT_DIR)