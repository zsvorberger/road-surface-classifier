import os
import random
import requests
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
TILE_CACHE = os.path.join(OUTPUT_DIR, "tile_cache")
DRAWN_DIR = os.path.join(OUTPUT_DIR, "DrawnTiles")

os.makedirs(TILE_CACHE, exist_ok=True)
os.makedirs(DRAWN_DIR, exist_ok=True)

NAIP_INDEX = os.path.join(OUTPUT_DIR, "naip_index.csv")
MERGED_LABELS = os.path.join(OUTPUT_DIR, "merged_labels.parquet")

# === LOAD DATA ===
print("📥 Loading NAIP index and merged labels...")
tiles = pd.read_csv(NAIP_INDEX)
labels = gpd.read_parquet(MERGED_LABELS)

# Make sure ID columns align
tiles["id"] = tiles["id"].astype(str)
labels["id"] = labels["id"].astype(str)

# Filter to valid tile URLs
tiles = tiles[tiles["tile_url"].notna() & (tiles["tile_url"].astype(str).str.strip() != "")]
merged = labels.merge(tiles, on="id", how="inner")

if merged.empty:
    print("❌ No overlapping IDs between NAIP index and merged_labels!")
    exit()

print(f"✅ Found {len(merged)} matching road entries")

# === PICK 5 RANDOM ROADS TO VISUALIZE ===
sample = merged.sample(n=min(5, len(merged)), random_state=42)

# === HELPER FUNCTION: DOWNLOAD TILE ===
def download_tile(url, save_path):
    """Download a tile from its signed URL if not already cached."""
    if os.path.exists(save_path):
        return True
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print(f"⬇️  Downloaded: {os.path.basename(save_path)}")
            return True
        else:
            print(f"⚠️ Failed download ({r.status_code}) for {url}")
            return False
    except Exception as e:
        print(f"❌ Download error for {url}: {e}")
        return False

# === MAIN LOOP ===
for idx, row in sample.iterrows():
    road_id = row["id"]
    tile_url = str(row["tile_url"]).strip()
    geom = row["geometry"]

    # Extract just the filename
    tile_filename = os.path.basename(tile_url.split("?")[0])
    tile_path = os.path.join(TILE_CACHE, tile_filename)

    # Ensure the tile is downloaded
    if not os.path.exists(tile_path):
        ok = download_tile(tile_url, tile_path)
        if not ok:
            print(f"⚠️ Skipping {road_id}: could not download tile")
            continue

    # Try plotting
    try:
        with rasterio.open(tile_path) as src:
            fig, ax = plt.subplots(figsize=(6, 6))
            show(src.read(), transform=src.transform, ax=ax)
            gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(src.crs).plot(ax=ax, color="red", linewidth=2)
            ax.set_title(f"Road {road_id}", fontsize=10)
            ax.axis("off")

            out_path = os.path.join(DRAWN_DIR, f"road_{road_id}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            print(f"✅ Saved overlay: {out_path}")

    except Exception as e:
        print(f"❌ Error processing {road_id}: {e}")

print("✅ Done! Check the DrawnTiles folder for output images.")