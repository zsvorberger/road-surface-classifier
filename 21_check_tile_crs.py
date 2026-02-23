from pathlib import Path
import pandas as pd
import rasterio
from pystac_client import Client
import planetary_computer

# --- Setup paths ---
BASE = Path(__file__).resolve().parents[1]
CSV_FILE = BASE / "output" / "naip_index.csv"

# --- Load and clean CSV ---
df = pd.read_csv(CSV_FILE, low_memory=False, on_bad_lines="skip")
df = df[pd.to_numeric(df["lon"], errors="coerce").notnull()]
df = df[pd.to_numeric(df["lat"], errors="coerce").notnull()]
df["lon"] = df["lon"].astype(float)
df["lat"] = df["lat"].astype(float)

print(f"Loaded {len(df)} clean rows from {CSV_FILE.name}")

# Pick one or two random tiles (or specify manually)
sample_rows = df.sample(2, random_state=42)

# --- Connect to Planetary Computer ---
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
print("✅ Connected to Planetary Computer")

for _, row in sample_rows.iterrows():
    lon, lat = float(row["lon"]), float(row["lat"])
    tile_id = row["tile_id"]
    print(f"\n=== Checking tile for road ID {row['id']} ===")
    print(f"Tile ID: {tile_id}")
    print(f"Coords: ({lon:.5f}, {lat:.5f})")

    # Search for the tile from Planetary Computer
    search = catalog.search(
        collections=["naip"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        query={"gsd": {"lt": 0.7}},
        limit=1,
        sortby=[{"field": "datetime", "direction": "desc"}],
    )

    items = list(search.get_all_items())
    if not items:
        print("⚠️ No items found for this coordinate.")
        continue

    item = items[0]
    href = list(item.assets.values())[0].href
    signed_href = planetary_computer.sign(href)

    print(f"Signed URL: {signed_href[:120]}...")

    # --- Open and inspect raster metadata ---
    try:
        with rasterio.open(signed_href) as src:
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Width x Height: {src.width} x {src.height}")
            print(f"Transform: {src.transform}")
            print(f"Pixel index for road location: {src.index(lon, lat)}")
    except Exception as e:
        print(f"❌ Error opening tile: {e}")
