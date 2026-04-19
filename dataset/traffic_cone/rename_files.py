#!/usr/bin/env python3
"""
rename_photos.py — Renames all photos in the same folder as this script.

Output format: abcdefghij.jpg, klmnopqrst.png, etc. (random 10-char lowercase letters)

Just drop this script into your photos folder and run:
    python rename_photos.py

Add --dry-run to preview without making any changes:
    python rename_photos.py --dry-run
"""

import argparse
import os
import random
import string
from pathlib import Path

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif", ".raw", ".cr2", ".nef", ".arw"}


def random_name(length: int = 10) -> str:
    """Generate a random lowercase letter string of the given length."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def unique_random_name(folder: Path, suffix: str, existing: set, length: int = 10) -> str:
    """Generate a random name that doesn't conflict with existing files or already-assigned names."""
    while True:
        name = random_name(length) + suffix.lower()
        if name not in existing and not (folder / name).exists():
            existing.add(name)
            return name


def rename_photos(dry_run: bool = False) -> None:
    folder = Path(__file__).parent.resolve()
    photos = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in PHOTO_EXTENSIONS]
    )

    if not photos:
        print("No photos found in this directory.")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Found {len(photos)} photo(s) in: {folder}\n")

    renamed, skipped = 0, 0
    assigned_names: set = set()

    for photo in photos:
        new_name = unique_random_name(folder, photo.suffix, assigned_names)
        new_path = folder / new_name

        if new_path == photo:
            print(f"  SKIP (unchanged): {photo.name}")
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
        description="Rename all photos in this script's directory to a random 10-character name."
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