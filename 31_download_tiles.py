"""
31_download_tiles_fast.py
-------------------------
Faster NAIP tile downloader — pulls real URLs from naip_index.csv,
downloads just 2 tiles for testing using multithreading + caching.
"""

import os
import requests
import pandas as pd
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# === CONFIG ===
MERGED_LABELS = "output/merged_labels.parquet"
NAIP_INDEX = "output/naip_index.csv"
CACHE_DIR = "output/tile_cache"
SAVE_DIR = "output/TestTiles"
NUM_SAMPLES = 2     # only 2 tiles to test speed + alignment
MAX_WORKERS = 4     # how many tiles to fetch in parallel

# Ensure folders exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Persistent HTTP session
session = requests.Session()

def download_tile(row):
    """Download a single tile (thread-safe)."""
    tile_url = str(row["tile_url"])
    tile_filename = os.path.basename(tile_url.split("?")[0])
    save_path = os.path.join(CACHE_DIR, tile_filename)

    if os.path.exists(save_path):
        print(f"🟡 Cached: {tile_filename}")
        return save_path

    try:
        with session.get(tile_url, stream=True, timeout=45) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):  # 512 KB chunks
                    f.write(chunk)
        print(f"⬇️  Downloaded: {tile_filename}")
        return save_path
    except Exception as e:
        print(f"❌ Failed {tile_filename}: {e}")
        return None


def visualize_tile(path):
    """Quickly save a preview image for manual checking."""
    try:
        with rasterio.open(path) as src:
            fig, ax = plt.subplots(figsize=(7, 7))
            show(src, ax=ax)
            ax.set_title(os.path.basename(path))
            out_png = os.path.join(SAVE_DIR, os.path.basename(path).replace(".tif", ".png"))
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"🖼️  Saved preview → {out_png}")
    except Exception as e:
        print(f"⚠️ Could not visualize {path}: {e}")


def main():
    print("=== Fast NAIP Tile Downloader ===")

    # Load index + labels
    tiles = pd.read_csv(NAIP_INDEX, low_memory=False)
    labels = gpd.read_parquet(MERGED_LABELS)

    tiles["id"] = tiles["id"].astype(str)
    labels["id"] = labels["id"].astype(str)
    merged = labels.merge(tiles[["id", "tile_url"]], on="id", how="inner")

    if merged.empty:
        print("❌ No overlap between NAIP index and labels.")
        return

    # Randomly pick 2 tiles to download
    sample = merged.sample(n=min(NUM_SAMPLES, len(merged)), random_state=42)

    print(f"Downloading {len(sample)} tiles using {MAX_WORKERS} threads...\n")

    # Parallel download
    downloaded = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_tile, row): row for _, row in sample.iterrows()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded.append(result)

    print(f"\n✅ {len(downloaded)} tiles ready in '{CACHE_DIR}'")

    # Save small preview PNGs
    for path in downloaded:
        visualize_tile(path)

    print("\n🟢 Done — check output/tile_cache for TIFFs and output/TestTiles for PNGs.")


if __name__ == "__main__":
    main()
