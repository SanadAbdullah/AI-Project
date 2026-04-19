"""
Bulk Image Downloader — Pixabay API
====================================
Downloads images reliably using the Pixabay API.

Setup:
    1. Get a FREE API key at https://pixabay.com/api/docs
    2. Paste your key into API_KEY below
    3. pip install requests
    4. python3 bulk_image_downloader.py

Size options: "webformatURL", "largeImageURL", "fullHDURL", "imageURL"
Rate limit:   100 requests/minute on free tier
"""

import requests
import os
import time

# ─────────────────────────────────────────────
#  ✏️  CONFIGURE YOUR DOWNLOAD HERE
# ─────────────────────────────────────────────
API_KEY       = "55399063-dd6fe95e5921855a0c3d62525"
KEYWORD       = "traffic cone"
NUM_IMAGES    = 110
OUTPUT_FOLDER = "traffic_cone"
IMAGE_SIZE    = "largeImageURL"   # "webformatURL", "largeImageURL", "fullHDURL", "imageURL"
# ─────────────────────────────────────────────

BASE_URL  = "https://pixabay.com/api/"
PER_PAGE  = 200  # Pixabay allows up to 200 hits per API request


def fetch_page(keyword, page, per_page, api_key):
    """Call Pixabay search API for one page of photo metadata; retry after cool-off on rate limit."""
    params = {
        "key":      api_key,
        "q":        keyword,
        "per_page": per_page,
        "page":     page,
        "image_type": "photo"
    }

    response = requests.get(BASE_URL, params=params, timeout=10)

    if response.status_code == 429:
        print("⏳ Rate limit reached. Waiting 60 seconds...")
        time.sleep(60)
        return fetch_page(keyword, page, per_page, api_key)

    if response.status_code != 200:
        print(f"❌ API error {response.status_code}: {response.text}")
        return None, 0

    data = response.json()
    total = data.get("totalHits", 0)
    return data.get("hits", []), total


def download_image(url, filepath):
    """Download binary image bytes from a direct URL and save to filepath."""
    response = requests.get(url, timeout=15)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
        return True
    return False


def run(keyword, num_images, output_folder, image_size, api_key):
    """Paginate Pixabay results until num_images are saved or results run out."""
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n🔍 Keyword:  {keyword}")
    print(f"📦 Target:   {num_images} images")
    print(f"📁 Folder:   {output_folder}/")
    print(f"🖼  Size:     {image_size}\n")

    downloaded = 0
    page = 1

    while downloaded < num_images:
        photos, total = fetch_page(keyword, page, PER_PAGE, api_key)

        if photos is None:
            break

        if not photos:
            print("⚠️  No results found for this keyword.")
            break

        for photo in photos:
            if downloaded >= num_images:
                break

            # Prefer configured size field; fall back so we still get an image if one key is missing
            url = photo.get(image_size) or photo.get("largeImageURL") or photo.get("webformatURL")

            if not url:
                print(f"  ⚠️  Skipped — no URL")
                continue

            # Infer file extension from URL path; strip query string; default to jpg if odd extension
            ext = url.split(".")[-1].split("?")[0]
            if ext.lower() not in ["jpg", "jpeg", "png", "webp"]:
                ext = "jpg"

            filepath = os.path.join(
                output_folder,
                f"{keyword.replace(' ', '_')}_{downloaded + 1:04d}.{ext}"
            )

            try:
                if download_image(url, filepath):
                    downloaded += 1
                    print(f"  ✅ [{downloaded}/{num_images}] {os.path.basename(filepath)}")
                else:
                    print(f"  ⚠️  Skipped — download failed")
            except Exception as e:
                print(f"  ⚠️  Skipped — {e}")

            # Tiny delay between file downloads to avoid hammering Pixabay/CDN
            time.sleep(0.05)

        # Stop if we've requested every page that could exist for totalHits
        if page * PER_PAGE >= total:
            print(f"\n⚠️  Exhausted all search results ({total} total found).")
            break

        page += 1

    print(f"\n🎉 Done! {downloaded} image(s) saved to '{output_folder}/'")


if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_PIXABAY_API_KEY_HERE":
        print("❌ Please add your Pixabay API key first.")
        print("   Get one free at: https://pixabay.com/api/docs")
    else:
        run(KEYWORD, NUM_IMAGES, OUTPUT_FOLDER, IMAGE_SIZE, API_KEY)