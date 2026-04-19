"""
Bulk Image Downloader — Pexels API
====================================
Downloads images reliably using the Pexels API.

Setup:
    1. Get a FREE API key at https://www.pexels.com/api/
    2. Paste your key into API_KEY below
    3. pip install requests
    4. python3 bulk_image_downloaderr.py
"""

import requests
import os
import time

# ─────────────────────────────────────────────
#  ✏️  CONFIGURE YOUR DOWNLOAD HERE
# ─────────────────────────────────────────────
API_KEY       = "fnjChFaSFFjfZ9Bwe2OHZMoTldzpp78N56A0oEwjFzMiPMHkG4QHS2mN"
KEYWORD       = "wet floor sign"
NUM_IMAGES    = 500
OUTPUT_FOLDER = "wet_floor_sign"
IMAGE_SIZE    = "large"   # options: "original", "large", "medium", "small"
# ─────────────────────────────────────────────

# Pexels: Authorization header is the raw API key (not "Bearer ...")


def download_images(keyword, num_images, output_folder, image_size, api_key):
    """
    Search Pexels page by page, download up to num_images photos into output_folder.
    Chooses URL from photo['src'][image_size] with fallbacks; stops on API error or no next_page.
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n📥 Downloading {num_images} images for: '{keyword}'")
    print(f"📁 Saving to: '{output_folder}/'\n")

    headers = {"Authorization": api_key}
    downloaded = 0
    page = 1
    per_page = min(80, num_images)  # Pexels allows at most 80 results per request

    while downloaded < num_images:
        params = {
            "query": keyword,
            "per_page": per_page,
            "page": page
        }

        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params
        )

        if response.status_code != 200:
            print(f"❌ API error {response.status_code}: {response.text}")
            break

        data = response.json()
        photos = data.get("photos", [])

        if not photos:
            print("⚠️  No more images found.")
            break

        for photo in photos:
            if downloaded >= num_images:
                break

            src = photo["src"]
            url = src.get(image_size) or src.get("large") or src.get("original")

            # Extension from URL path; strip query string first
            ext = url.split("?")[0].split(".")[-1]
            if ext.lower() not in ["jpg", "jpeg", "png", "webp"]:
                ext = "jpg"

            filename = os.path.join(output_folder, f"{keyword.replace(' ', '_')}_{downloaded + 1:04d}.{ext}")

            try:
                img_data = requests.get(url, timeout=10)
                if img_data.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(img_data.content)
                    downloaded += 1
                    print(f"  ✅ [{downloaded}/{num_images}] Saved {os.path.basename(filename)}")
                else:
                    print(f"  ⚠️  Skipped (HTTP {img_data.status_code})")
            except Exception as e:
                print(f"  ⚠️  Skipped ({e})")

            time.sleep(0.05)

        page += 1

        if not data.get("next_page"):
            print("⚠️  Reached end of search results.")
            break

    print(f"\n🎉 Done! {downloaded} images saved to '{output_folder}/'")


if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_PEXELS_API_KEY_HERE":
        print("❌ Please add your Pexels API key to the script first.")
        print("   Get one free at: https://www.pexels.com/api/")
    else:
        download_images(KEYWORD, NUM_IMAGES, OUTPUT_FOLDER, IMAGE_SIZE, API_KEY)