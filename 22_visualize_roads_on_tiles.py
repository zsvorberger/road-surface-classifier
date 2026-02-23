import os
import random
import requests
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# === CONFIG ===
MERGED_LABELS = "output/merged_labels.parquet"
NAIP_INDEX = "output/naip_index.csv"
SAVE_DIR = "output/DrawnTiles"
CACHE_DIR = "output/tile_cache"
NUM_SAMPLES = 2           # only draw a couple for testing
ROAD_WIDTH_METERS = 8     # exact visible width for classifier (approx. full road)
LINE_COLOR = "red"        # solid line, no transparency

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def download_tile(url, save_path):
    if os.path.exists(save_path):
        return True
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"⬇️  Downloaded {os.path.basename(save_path)}")
        return True
    except Exception as e:
        print(f"❌ Download error for {url}: {e}")
        return False

def main():
    print("=== Drawing solid red road overlays (exact classifier width) ===")

    # Load datasets
    tiles = pd.read_csv(NAIP_INDEX)
    labels = gpd.read_parquet(MERGED_LABELS)

    tiles["id"] = tiles["id"].astype(str)
    labels["id"] = labels["id"].astype(str)

    # Merge
    merged = labels.merge(
        tiles[["id", "tile_url"]],
        on="id",
        how="inner"
    )

    if merged.empty:
        print("❌ No overlapping IDs between datasets.")
        return

    sample = merged.sample(n=min(NUM_SAMPLES, len(merged)), random_state=42)

    for _, row in sample.iterrows():
        road_id = row["id"]
        geom = row["geometry"]
        tile_url = str(row["tile_url"])
        tile_filename = os.path.basename(tile_url.split("?")[0])
        tile_path = os.path.join(CACHE_DIR, tile_filename)

        if not os.path.exists(tile_path):
            ok = download_tile(tile_url, tile_path)
            if not ok:
                continue

        try:
            with rasterio.open(tile_path) as src:
                fig, ax = plt.subplots(figsize=(10, 10))
                show(src, ax=ax)

                # Convert road to raster CRS
                road_gdf = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(src.crs)

                # Create a road strip ~8 m wide
                road_strip = road_gdf.buffer(ROAD_WIDTH_METERS)

                # Draw it solid red (no transparency)
                road_strip.plot(ax=ax, color=LINE_COLOR, alpha=1.0, edgecolor="none")

                ax.set_title(f"Road {road_id} (width ≈ {ROAD_WIDTH_METERS} m)", fontsize=11)
                ax.axis("off")

                out_path = os.path.join(SAVE_DIR, f"road_{road_id}_solid.png")
                fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"❌ Error on {road_id}: {e}")

    print("\n✅ Done — check DrawnTiles for crisp solid road overlays.")

if __name__ == "__main__":
    main()
