#!/usr/bin/env python3
"""
Rename photos in this script's folder using date + sequence numbers.

Output format: YYYY-MM-DD_001.jpg, YYYY-MM-DD_002.png, etc.
The date is each file's last-modified date (not EXIF). Numbers restart at 001
per run and follow the sorted list order (not chronological photo order).

Usage:
    python "rename_files 2.py"
    python "rename_files 2.py" --dry-run
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

# Same extension list as rename_files.py — only these files are renamed
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif", ".raw", ".cr2", ".nef", ".arw"}


def get_photo_date(file: Path) -> str:
    """Use the file's last-modified date, formatted as YYYY-MM-DD."""
    mtime = file.stat().st_mtime
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


def rename_photos(dry_run: bool = False) -> None:
    """Rename each photo to YYYY-MM-DD_NNN.ext using mtime date and enumerate order."""
    folder = Path(__file__).parent.resolve()
    photos = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in PHOTO_EXTENSIONS]
    )

    if not photos:
        print("No photos found in this directory.")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Found {len(photos)} photo(s) in: {folder}\n")

    renamed, skipped = 0, 0

    for i, photo in enumerate(photos, start=1):
        date_str = get_photo_date(photo)
        # Sequence i is position in sorted directory listing, not “same calendar day” grouping
        new_name = f"{date_str}_{i:03d}{photo.suffix.lower()}"
        new_path = folder / new_name

        if new_path == photo:
            print(f"  SKIP (unchanged): {photo.name}")
            skipped += 1
            continue

        if new_path.exists():
            print(f"  SKIP (conflict):  {photo.name} → {new_name} already exists")
            skipped += 1
            continue

        print(f"  {'WOULD RENAME' if dry_run else 'RENAME'}: {photo.name}  →  {new_name}")

        if not dry_run:
            photo.rename(new_path)
            renamed += 1
        else:
            renamed += 1

    print(f"\nDone — {renamed} renamed, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename all photos in this script's directory to YYYY-MM-DD_001.ext format."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without renaming anything",
    )
    args = parser.parse_args()
    rename_photos(dry_run=args.dry_run)


if __name__ == "__main__":
    main()